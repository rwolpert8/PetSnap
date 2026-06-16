# Models

This folder contains the trained model weights for the PetSnap dog breed classifier.

## Contents

- **`best_model.pth`** - ResNet101 model trained on Stanford Dogs Dataset
  - Architecture: ResNet101 (pretrained on ImageNet, fine-tuned)
  - Classes: 120 dog breeds
  - Size: ~170 MB
  - Input: 224x224 RGB images
  - Output: 120-class probability distribution

## Training

The model is trained using [PetSnap_NN.ipynb](../src/PetSnap_NN.ipynb) and automatically saved to this folder during training.

## Usage

The model is loaded by the API server in `api/api_server.py`:

```python
model.load_state_dict(torch.load('../models/best_model.pth', map_location=device))
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

This folder is excluded from git version control (see `.gitignore`) to avoid committing large model files to the repository.
