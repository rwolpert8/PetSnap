"""Integration checks use the real local checkpoint; no training dataset required."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from api import api_server
from api.inference import BASE_DIR, MAX_UPLOAD_BYTES, InvalidImage, decode_image, load_classes


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Startup must work even when dataset discovery is explicitly unavailable.
        cls.dataset_guard = patch("torchvision.datasets.ImageFolder", side_effect=AssertionError("Dataset accessed"))
        cls.dataset_guard.start()
        cls.client = TestClient(api_server.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.dataset_guard.stop()

    def test_web_and_health(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="drop-zone"', response.text)
        self.assertEqual(self.client.get("/health").json(), {"status": "ready", "classes": 120})
        self.assertEqual(self.client.get("/models/best_model.pth").status_code, 404)

    def test_real_predictions(self):
        for filename, expected in [("golden", "Golden Retriever"), ("beagle", "Beagle"), ("husky", "Siberian Husky")]:
            with self.subTest(filename=filename):
                data = (BASE_DIR / f"frontend/assets/{filename}.jpg").read_bytes()
                response = self.client.post("/api/demo/predict", files={"file": ("dog.jpg", data, "image/jpeg")})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["predicted_breed"], expected)
                predictions = result["top_5_predictions"]
                self.assertEqual(len(predictions), 5)
                scores = [item["confidence"] for item in predictions]
                self.assertEqual(scores, sorted(scores, reverse=True))
                self.assertTrue(all(0 <= score <= 100 for score in scores))

    def test_invalid_images_are_client_errors(self):
        for data in [b"not an image", b""]:
            response = self.client.post("/api/demo/predict", files={"file": ("dog.jpg", data, "image/jpeg")})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.post("/api/demo/predict").status_code, 422)

    def test_oversize_upload(self):
        response = self.client.post("/api/demo/predict", files={"file": ("dog.jpg", b"x" * (MAX_UPLOAD_BYTES + 1), "image/jpeg")})
        self.assertEqual(response.status_code, 413)

    def test_concurrent_inference_has_retry_response(self):
        with api_server.app.state.classifier._lock:
            response = self.client.post("/api/demo/predict", files={"file": ("dog.jpg", b"small", "image/jpeg")})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["retry-after"], "2")

    def test_private_endpoints_require_key(self):
        for endpoint in ["/predict", "/predict_batch", "/api/identify", "/identify"]:
            self.assertEqual(self.client.post(endpoint).status_code, 401)

    def test_explicit_key_preserves_predict_contract(self):
        data = (BASE_DIR / "frontend/assets/golden.jpg").read_bytes()
        with patch.object(api_server, "API_KEY", "integration-test-key"):
            response = self.client.post("/predict", headers={"X-API-Key": "integration-test-key"},
                                        files={"file": ("dog.jpg", data, "image/jpeg")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["predicted_breed"], "Golden Retriever")


class ImageAndLabelTests(unittest.TestCase):
    def test_invalid_label_manifest_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classes.json"
            for labels in [[], ["same"] * 120, [None] * 120]:
                path.write_text(json.dumps(labels), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_classes(path)

    def test_orientation_and_grayscale(self):
        source = Image.new("L", (30, 20))
        exif = source.getexif()
        exif[274] = 6
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", exif=exif)
        decoded = decode_image(buffer.getvalue())
        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.size, (20, 30))

    def test_unsupported_image_format(self):
        buffer = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buffer, format="GIF")
        with self.assertRaises(InvalidImage):
            decode_image(buffer.getvalue())

    def test_pixel_limit(self):
        buffer = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buffer, format="PNG")
        with patch("api.inference.MAX_IMAGE_PIXELS", 9999):
            with self.assertRaises(InvalidImage):
                decode_image(buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
