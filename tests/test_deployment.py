import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deploy.download_model import ModelDownloadError, fetch_model, verify_labels, main


class Body(io.BytesIO):
    def iter_chunks(self, chunk_size):
        while chunk := self.read(3):
            yield chunk


class Client:
    def __init__(self, data, size=None):
        self.body = Body(data)
        self.size = len(data) if size is None else size

    def get_object(self, **kwargs):
        self.arguments = kwargs
        return {"Body": self.body, "ContentLength": self.size}


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.target = Path(self.directory.name) / "model.pth"
        self.data = b"a private model checkpoint"
        self.manifest = {"size_bytes": len(self.data), "sha256": hashlib.sha256(self.data).hexdigest()}

    def test_verified_download_publishes_atomically(self):
        client = Client(self.data)
        fetch_model(client, "private", "v1/model", self.target, self.manifest, "version-1")
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertEqual(client.arguments["VersionId"], "version-1")
        self.assertTrue(client.body.closed)
        self.assertEqual(list(self.target.parent.glob("*.part")), [])

    def test_hash_mismatch_preserves_previous_file(self):
        self.target.write_bytes(b"previous checkpoint")
        client = Client(b"x" * len(self.data))
        with self.assertRaises(ModelDownloadError):
            fetch_model(client, "private", "key", self.target, self.manifest)
        self.assertEqual(self.target.read_bytes(), b"previous checkpoint")
        self.assertTrue(client.body.closed)
        self.assertEqual(list(self.target.parent.glob("*.part")), [])

    def test_truncated_and_oversize_streams_are_rejected(self):
        for data in [self.data[:-1], self.data + b"extra"]:
            with self.subTest(length=len(data)):
                client = Client(data, len(self.data))
                with self.assertRaises(ModelDownloadError):
                    fetch_model(client, "private", "key", self.target, self.manifest)
                self.assertFalse(self.target.exists())
                self.assertTrue(client.body.closed)
                self.assertEqual(list(self.target.parent.glob("*.part")), [])

    def test_wrong_declared_size_is_rejected(self):
        client = Client(self.data, 999)
        with self.assertRaises(ModelDownloadError):
            fetch_model(client, "private", "key", self.target, self.manifest)
        self.assertTrue(client.body.closed)
        self.assertFalse(self.target.exists())

    def test_download_deadline_cleans_partial_file(self):
        client = Client(self.data)
        with patch("deploy.download_model.time.monotonic", side_effect=[0, 301]):
            with self.assertRaises(ModelDownloadError):
                fetch_model(client, "private", "key", self.target, self.manifest)
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.target.parent.glob("*.part")), [])

    def test_label_fingerprint_ignores_formatting_but_detects_reordering(self):
        labels = ["Beagle", "Pug"]
        expected = hashlib.sha256(json.dumps(labels, separators=(",", ":")).encode()).hexdigest()
        path = self.target.parent / "labels.json"
        path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        verify_labels(path, expected)
        path.write_text(json.dumps(list(reversed(labels))), encoding="utf-8")
        with self.assertRaises(ModelDownloadError):
            verify_labels(path, expected)

    def test_provider_errors_do_not_expose_private_details(self):
        with patch("deploy.download_model.ensure_model", side_effect=RuntimeError("secret-access-key")):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertNotIn("secret-access-key", str(error.exception))

    def test_cached_model_needs_no_storage_credentials(self):
        from deploy.download_model import ensure_model
        self.target.write_bytes(self.data)
        root = self.target.parent
        (root / "deploy").mkdir()
        (root / "models").mkdir()
        labels = root / "models/classes.json"
        labels.write_text('["Beagle"]', encoding="utf-8")
        self.manifest["labels_sha256"] = hashlib.sha256(b'["Beagle"]').hexdigest()
        (root / "deploy/model-manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        with patch("deploy.download_model.ROOT", root), patch.dict("os.environ", {
                "PETSNAP_MODEL_PATH": str(self.target), "PETSNAP_CLASSES_PATH": str(labels)}, clear=True):
            self.assertEqual(ensure_model(), self.target)

    def test_startup_uses_single_worker_and_runtime_port(self):
        from deploy import start
        with patch.dict("os.environ", {"PORT": "8018"}, clear=True), \
                patch.object(start, "prepare_model", return_value=self.target), \
                patch.object(start.os, "chdir"), patch.object(start.os, "execv") as launch:
            start.main()
        args = launch.call_args.args[1]
        self.assertEqual(args[args.index("--workers") + 1], "1")
        self.assertEqual(args[args.index("--port") + 1], "8018")
        self.assertIn("--no-proxy-headers", args)

    def test_invalid_startup_settings_fail_before_download(self):
        from deploy import start
        for env in [{"PORT": "0"}, {"PORT": "not-a-port"}, {"PETSNAP_TRUSTED_PROXY_IPS": "*"}]:
            with patch.dict("os.environ", env, clear=True), patch.object(start.os, "chdir"), \
                    patch.object(start, "prepare_model") as download:
                with self.assertRaises(SystemExit):
                    start.main()
                download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
