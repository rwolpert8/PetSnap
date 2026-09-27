# PetSnap - AI-Powered Dog Breed Recognition

PetSnap is a deep learning-based dog breed recognition system that identifies dog breeds from images with high accuracy. The system uses a fine-tuned ResNet101 model trained on the Stanford Dogs dataset and provides detailed breed information scraped from the American Kennel Club (AKC) website.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **High Accuracy Recognition**: Fine-tuned ResNet101 model achieving strong performance on 120 dog breeds
- **RESTful API**: Fast and secure FastAPI server with authentication
- **Breed Information**: Automatically scrapes detailed breed information from AKC website
- **Top-5 Predictions**: Returns top 5 most likely breeds with confidence scores
- **Batch Processing**: Support for processing multiple images simultaneously
- **Test Time Augmentation**: Enhanced prediction accuracy using TTA techniques
- **Mobile-Ready**: CORS-enabled endpoints compatible with mobile applications

## Model Performance

- **Architecture**: ResNet101 (pretrained on ImageNet)
- **Dataset**: Stanford Dogs Dataset (20,580 images, 120 breeds)
- **Training Set**: 16,464 images (80%)
- **Validation Set**: 4,116 images (20%)
- **Overall Accuracy**: 90.04%
- **Top-3 Accuracy**: 98.25%
- **Top-5 Accuracy**: 99.27%
- **Optimization**: AdamW optimizer with cosine annealing warm restarts
- **Training Techniques**:
  - Label smoothing (0.1)
  - AutoAugment data augmentation
  - Mixed precision training
  - Early stopping with patience
  - Test Time Augmentation for inference

See [MODEL_PERFORMANCE.md](docs/MODEL_PERFORMANCE.md) for detailed metrics and confusion matrix.

## Architecture

```
User → Mobile/Web App → FastAPI Server → ResNet101 Model
                              ↓
                        AKC Web Scraper
                              ↓
                     Breed Information
```

For detailed architecture information, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Run the Web Demo

The new responsive browser demo includes photo upload, image preview, three
sample dogs, and five ranked breed predictions. FastAPI serves the interface
and model from a single process; no frontend build or API key is needed.

With Python 3.10+ and the trained checkpoint at `models/best_model.pth`:

```bash
python -m pip install -r requirements-inference.txt
python -m uvicorn api.api_server:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. The local checkpoint is excluded from Git, so a
fresh clone needs a copy of your trained weights. Keep the tracked
`models/classes.json` beside it. **The demo does not require the training
dataset or retraining.**

See [frontend/README.md](frontend/README.md) for setup, configuration, tests,
photo credits, and the remaining public-hosting steps. The existing Expo
mobile project remains separate in `mobile-app/`.

To train a new model, install `requirements.txt`, obtain the Stanford Dogs
dataset in `Images/`, and run `src/PetSnap_NN.ipynb`. The notebook saves the
ordered class labels alongside every best checkpoint.

## API Documentation

For Docker packaging and Render hosting with private weights, see
[the deployment walkthrough](deploy/README.md). Cloud storage and hosting have
not been provisioned by these deployment files.

### Authentication

The browser uses the public `POST /api/demo/predict` endpoint. Legacy prediction
endpoints require `DOG_CLASSIFIER_API_KEY` to be explicitly configured on the
server; there is no built-in key. Include it in the request header:

```
X-API-Key: your_api_key_here
```

### Endpoints

#### **GET /health** - Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ready",
  "classes": 120
}
```

#### **GET /classes** - Get All Breeds
```bash
curl http://localhost:8000/classes
```

Response:
```json
{
  "classes": ["Chihuahua", "Japanese spaniel", ...],
  "total_classes": 120
}
```

#### **POST /predict** - Predict Dog Breed
```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your_api_key" \
  -F "file=@dog_image.jpg"
```

Response:
```json
{
  "success": true,
  "predicted_breed": "golden_retriever",
  "confidence": 95.67,
  "top_5_predictions": [
    {"breed": "golden_retriever", "confidence": 95.67},
    {"breed": "Labrador_retriever", "confidence": 3.21},
    {"breed": "flat-coated_retriever", "confidence": 0.89},
    {"breed": "Chesapeake_Bay_retriever", "confidence": 0.15},
    {"breed": "curly-coated_retriever", "confidence": 0.08}
  ]
}
```

#### **POST /api/identify** - Identify with Breed Info
```bash
curl -X POST http://localhost:8000/api/identify \
  -H "X-API-Key: your_api_key" \
  -F "file=@dog_image.jpg"
```

Response:
```json
{
  "status": "success",
  "result": {
    "breed": "golden_retriever",
    "confidence": 95.67,
    "alternatives": [...]
  }
}
```

#### **POST /identify** - Identify from Base64 (Mobile-Friendly)
```bash
curl -X POST http://localhost:8000/identify \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"imageDataUrl": "data:image/jpeg;base64,/9j/4AAQ..."}'
```

Response:
```json
{
  "animalName": "golden_retriever",
  "description": "The Golden Retriever is a friendly, intelligent...",
  "confidence": "Identified with 95.7% confidence",
  "learnMoreUrl": "https://www.akc.org/dog-breeds/golden-retriever/",
  "funFacts": ["This breed was identified with 95.7% confidence."],
  "sources": [...]
}
```

#### **POST /predict_batch** - Batch Prediction (Max 10 images)
```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "X-API-Key: your_api_key" \
  -F "files=@dog1.jpg" \
  -F "files=@dog2.jpg"
```

### Interactive API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Model Training

The model training process is documented in `src/PetSnap_NN.ipynb`. Key features:

1. **Data Augmentation**
   - AutoAugment (ImageNet policy)
   - Random resized crop
   - Random horizontal flip
   - Color jitter

2. **Training Configuration**
   - Batch size: 96
   - Learning rate: 0.0003-0.0005
   - Weight decay: 1e-4
   - Label smoothing: 0.1
   - Epochs: 25 (with early stopping)

3. **Optimization Techniques**
   - Mixed precision training (AMP)
   - Cosine annealing warm restarts
   - Gradient scaling
   - Early stopping (patience: 5)

4. **Inference Enhancement**
   - Test Time Augmentation (TTA)
   - Top-K predictions

## Project Structure

```
PetSnap/
├── requirements.txt                # Python dependencies
├── LICENSE.md                      # MIT License
├── README.md                       # This file
├── api/                            # API-related code and modules
│   └── api_server.py               # FastAPI server implementation
├── src/                            # Source code and utilities
│   └── PetSnap_NN.ipynb            # Model training notebook
├── frontend/                       # Responsive browser demo, served by FastAPI
├── models/                         # Trained model weights
│   └── best_model.pth              # ResNet101 trained model
├── results/                        # Training results and performance visualizations
│   ├── generate_performance_metrics.py  # Performance metrics script
│   ├── confusion_matrix_full.png   # Generated visualizations
│   ├── confusion_matrix_normalized.png
│   ├── per_class_accuracy.png      # Per-breed accuracy chart
│   └── ...                         # Additional metrics and outputs
├── Images/                         # Stanford Dogs Dataset (120 breed folders)
└── docs/                           # Documentation
    ├── ARCHITECTURE.md             # System architecture
    ├── DESIGN_DECISIONS.md         # Design rationale
    ├── FUTURE_WORK.md              # Roadmap
    └── MODEL_PERFORMANCE.md        # Performance analysis
```

## Configuration

### Environment Variables

- `DOG_CLASSIFIER_API_KEY`: Required for legacy authenticated API routes; no default.
- `PETSNAP_MODEL_PATH`: Optional path to the trained checkpoint.
- `PETSNAP_CLASSES_PATH`: Optional path to its ordered labels JSON.
- `PETSNAP_CORS_ORIGINS`: Optional comma-separated allowed origins for separate clients.

### Model Configuration

See `api/inference.py` for:
- Model architecture
- Input image size
- Normalization parameters
- Device selection (CPU/GPU)

## Supported Dog Breeds

PetSnap can identify 120 different dog breeds from the Stanford Dogs dataset, including:

- **Toy Breeds**: Chihuahua, Pomeranian, Pug, Shih-Tzu, Yorkshire Terrier
- **Sporting Dogs**: Golden Retriever, Labrador Retriever, Cocker Spaniel
- **Working Dogs**: Siberian Husky, Great Dane, Rottweiler, Doberman
- **Herding Dogs**: Border Collie, German Shepherd, Australian Shepherd
- **And many more!**

See `/classes` endpoint for the complete list.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Stanford Dogs Dataset**: Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Li Fei-Fei
- **ResNet Architecture**: Kaiming He et al.
- **American Kennel Club**: For breed information and educational content
- **PyTorch Team**: For the excellent deep learning framework
- **FastAPI**: For the modern, fast web framework

## Future Work

See [FUTURE_WORK.md](docs/FUTURE_WORK.md) for planned features and improvements.

---
