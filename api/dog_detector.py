"""Whole-photo dog detection using Torchvision's public COCO checkpoint."""
import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

WEIGHTS = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
# Initial operating threshold, not a calibrated probability. See docs/dog-detection.md.
DOG_THRESHOLD = 0.5


class DogDetector:
    def __init__(self, device):
        self.device = device
        self.dog_label = WEIGHTS.meta["categories"].index("dog")
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=None, num_classes=len(WEIGHTS.meta["categories"]),
            min_size=480, max_size=640)
        self.model.load_state_dict(WEIGHTS.get_state_dict(progress=False, check_hash=True))
        self.model.to(device).eval()
        self.transform = WEIGHTS.transforms()

    def contains_dog(self, image):
        # Preserve the complete frame and bound tensor allocation before transforms.
        preview = image.copy()
        preview.thumbnail((640, 640), Image.Resampling.LANCZOS)
        with torch.inference_mode():
            result = self.model([self.transform(preview).to(self.device)])[0]
        return bool(((result["labels"] == self.dog_label)
                     & (result["scores"] >= DOG_THRESHOLD)).any().item())


if __name__ == "__main__":
    # Populate the public checkpoint cache during Docker build, not on uploads.
    WEIGHTS.get_state_dict(progress=True, check_hash=True)
