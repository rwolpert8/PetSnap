"""Dataset-independent inference. Deploy the weights and their ordered labels together."""

import io
import json
import os
import threading
import warnings
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent.parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImage(ValueError):
    pass


class InferenceBusy(RuntimeError):
    pass


def load_classes(path):
    classes = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(classes, list) or len(classes) != 120
            or not all(isinstance(label, str) and label.strip() for label in classes)
            or len(set(classes)) != 120):
        raise ValueError("classes.json must contain exactly 120 unique, ordered breed names")
    return classes


def decode_image(data):
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImage("Choose an image smaller than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in ALLOWED_FORMATS:
                    raise InvalidImage("Choose a JPG, PNG, or WebP image.")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise InvalidImage("Choose an image with fewer than 20 million pixels.")
                return ImageOps.exif_transpose(source).convert("RGB")
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImage("This image could not be opened. Try a JPG, PNG, or WebP photo.") from exc


class Classifier:
    def __init__(self, model_path=None, classes_path=None):
        model_path = Path(model_path or os.getenv("PETSNAP_MODEL_PATH", BASE_DIR / "models/best_model.pth"))
        classes_path = Path(classes_path or os.getenv("PETSNAP_CLASSES_PATH", BASE_DIR / "models/classes.json"))
        self.classes = load_classes(classes_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing trained weights: {model_path}. See models/README.md.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet101(weights=None, num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._lock = threading.Lock()

    def predict(self, data):
        # Bound decoding and inference together so concurrent uploads cannot exhaust RAM.
        if not self._lock.acquire(blocking=False):
            raise InferenceBusy("PetSnap is meeting another dog. Try again in a moment.")
        try:
            image = decode_image(data)
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                probabilities = self.model(tensor).softmax(dim=1)
                scores, indices = probabilities.topk(5, dim=1)
            predictions = [
                {"breed": self.classes[index], "confidence": round(score * 100, 2)}
                for score, index in zip(scores[0].tolist(), indices[0].tolist())
            ]
            return {"success": True, "predicted_breed": predictions[0]["breed"],
                    "confidence": predictions[0]["confidence"], "top_5_predictions": predictions}
        finally:
            self._lock.release()
