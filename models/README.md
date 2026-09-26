# Models

This folder contains the trained model weights for the PetSnap dog breed classifier.

## Contents

- **`best_model.pth`** - ResNet101 model trained on Stanford Dogs Dataset
  - Architecture: ResNet101 (pretrained on ImageNet, fine-tuned)
  - Classes: 120 dog breeds
  - Size: ~170 MB
  - Input: 224x224 RGB images
  - Output: 120-class probability distribution
- **`classes.json`** - The 120 breed names in model-output index order. This file
  is tracked in Git and required at startup. It was exported from the current
  dataset folder names using the same case-sensitive sorting as `ImageFolder`.

## Training

The model is trained using [PetSnap_NN.ipynb](../src/PetSnap_NN.ipynb). Each best
checkpoint save now also writes `full_dataset.classes` to `classes.json`.
Keep the checkpoint and this label file together; changing label order silently
changes the meaning of predictions. Existing weights cannot reveal breed names
on their own, so do not regenerate this mapping from a different dataset layout.

## Usage

The API uses `api/inference.py` to load the model once at startup:

```python
from api.inference import Classifier

classifier = Classifier()
prediction = classifier.predict(open('dog.jpg', 'rb').read())
```

## Model Details

- **Base Architecture**: ResNet101
- **Pretrained**: ImageNet weights
- **Fine-tuned on**: Stanford Dogs Dataset (20,580 images, 120 breeds)
- **Training Configuration**:
  - Optimizer: AdamW
  - Scheduler: Cosine Annealing with Warm Restarts
  - Loss: Cross-Entropy with Label Smoothing (0.1)
  - Data Augmentation: AutoAugment, RandomResizedCrop, ColorJitter
  - Mixed Precision Training (AMP)
  - Early Stopping (patience=5)

## Git Ignore

Only weight files are excluded from Git (see `.gitignore`); documentation and
`classes.json` are tracked. A deployment needs the weights and labels, not the
`Images/` training dataset. Paths can be overridden with `PETSNAP_MODEL_PATH`
and `PETSNAP_CLASSES_PATH`. Missing or invalid labels fail startup instead of
falling back to a potentially incorrect ordering.
