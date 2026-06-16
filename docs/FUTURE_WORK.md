# Future Work - PetSnap

This document outlines planned improvements, feature additions, and research directions for the PetSnap project. Items are organized by priority and complexity.

## Table of Contents
- [High Priority Improvements](#high-priority-improvements)
- [Model Enhancements](#model-enhancements)
- [API & Infrastructure](#api--infrastructure)
- [User Experience](#user-experience)
- [Research Directions](#research-directions)
- [Technical Debt](#technical-debt)

---

## High Priority Improvements

### 1. Model Performance Optimization 

#### 1.1 Model Quantization
**Goal**: Reduce model size and inference latency

**Approach**:
- INT8 quantization using PyTorch's quantization tools
- Post-training static quantization (PTQ)
- Quantization-aware training (QAT) for better accuracy retention

**Expected Benefits**:
- 4× smaller model size (~170MB → ~45MB)
- 2-3× faster inference on CPU
- Lower GPU memory usage

**Effort**: Medium (2-3 weeks)

**Implementation Plan**:
```python
# Post-training quantization
import torch.quantization as quantization
model_int8 = quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)
```

#### 1.2 ONNX Export for Cross-Platform Deployment
**Goal**: Deploy model to mobile, web, and edge devices

**Approach**:
- Export PyTorch model to ONNX format
- Optimize with ONNX Runtime
- Test on mobile (ONNX Mobile) and web (ONNX.js)

**Expected Benefits**:
- 30-50% faster inference with ONNX Runtime
- Run directly in browser with ONNX.js
- Deploy to mobile with ONNX Mobile

**Effort**: Medium (2-3 weeks)

#### 1.3 Model Distillation
**Goal**: Create smaller "student" model that mimics ResNet101

**Approach**:
- Train EfficientNet-B0 or MobileNetV3 as student
- Use ResNet101 predictions as soft targets
- Knowledge distillation loss

**Expected Benefits**:
- 10-20× smaller model
- 5-10× faster inference
- ~3-5% accuracy drop (acceptable trade-off)

**Effort**: High (4-6 weeks)

---

### 2. Breed Information Caching 

**Goal**: Reduce AKC scraping overhead and improve response time

**Current Issue**:
- Every prediction scrapes AKC website (~1-3s)
- Fragile to website changes
- Rate limiting concerns

**Proposed Solution**:

#### Option A: In-Memory Cache (Quick Win)
```python
from functools import lru_cache

@lru_cache(maxsize=120)  # Cache all 120 breeds
def get_akc_breed_info(breed_name):
    # Existing implementation
    pass
```

**Effort**: Low (1 day)  
**Benefits**: Instant responses for cached breeds

#### Option B: Redis Cache (Production)
```python
import redis

cache = redis.Redis(host='localhost', port=6379, db=0)

def get_akc_breed_info(breed_name):
    cached = cache.get(f"breed:{breed_name}")
    if cached:
        return json.loads(cached)
    
    info = scrape_akc(breed_name)
    cache.setex(f"breed:{breed_name}", 86400, json.dumps(info))  # 24h TTL
    return info
```

**Effort**: Medium (1 week)  
**Benefits**: Persistent cache, scales horizontally

#### Option C: Pre-scraped Database
- Scrape all 120 breeds offline
- Store in SQLite/PostgreSQL
- Update monthly via scheduled job

**Effort**: Medium (1-2 weeks)  
**Benefits**: No runtime scraping, 100% reliable

**Recommendation**: Start with Option A, move to Option B for production

---

### 3. Comprehensive Testing Suite 

**Goal**: Ensure reliability and catch regressions

**Test Categories**:

#### Unit Tests
- Model loading and inference
- Image preprocessing
- API key validation
- Breed name mapping

#### Integration Tests
- End-to-end API requests
- Multi-image batch processing
- Error handling and edge cases

#### Performance Tests
- Latency benchmarks (p50, p95, p99)
- Throughput under load
- Memory leak detection

**Tools**:
- `pytest` for test framework
- `pytest-asyncio` for async tests
- `locust` for load testing

**Effort**: Medium (2-3 weeks)

**Example**:
```python
def test_predict_golden_retriever():
    with open("test_images/golden_retriever.jpg", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": f},
            headers={"X-API-Key": "test_key"}
        )
    assert response.status_code == 200
    assert "golden_retriever" in response.json()["predicted_breed"].lower()
    assert response.json()["confidence"] > 80.0
```

**Coverage Goal**: >80% code coverage

---

## Model Enhancements

### 4. Expand to More Breeds 

**Current**: 120 breeds (Stanford Dogs)  
**Goal**: 200+ breeds (AKC-recognized)

**Approach**:
- Scrape additional breed images from web
- Use data augmentation to balance dataset
- Fine-tune existing model on expanded dataset

**Challenges**:
- Data collection and labeling
- Class imbalance for rare breeds
- Maintaining accuracy on existing breeds

**Effort**: High (6-8 weeks)

**Data Sources**:
- Google Images (with legal review)
- Flickr Creative Commons
- User-contributed images (with consent)

---

### 5. Multi-Dog Detection 

**Goal**: Detect and classify multiple dogs in single image

**Approach**:

#### Phase 1: Object Detection
- Add YOLO or Faster R-CNN for dog detection
- Crop each detected dog
- Run breed classification on each crop

#### Phase 2: End-to-End Model
- Train unified detection + classification model
- Return bounding boxes + breed labels

**Expected Output**:
```json
{
  "dogs_detected": 2,
  "predictions": [
    {
      "bbox": [100, 50, 300, 400],
      "breed": "golden_retriever",
      "confidence": 94.2
    },
    {
      "bbox": [350, 100, 550, 450],
      "breed": "labrador_retriever",
      "confidence": 89.7
    }
  ]
}
```

**Effort**: Very High (8-12 weeks)

---

### 6. Age and Gender Prediction 

**Goal**: Predict dog's approximate age and gender

**Approach**:
- Multi-task learning: Add age/gender heads to model
- Requires labeled dataset (challenging to obtain)
- Age as regression (0-15 years), gender as binary classification

**Model Architecture**:
```
ResNet101 Backbone
        ↓
    Features (2048)
    ↙     ↓     ↘
Breed   Age   Gender
(120)   (1)   (2)
```

**Challenges**:
- Difficult to determine age from photos
- Gender not visible for many breeds (neutered/spayed)
- Requires expert labeling

**Effort**: Very High (12+ weeks)

---

### 7. Similar Breeds Explanation

**Goal**: Explain why model might confuse similar breeds

**Approach**:
- Feature visualization (Grad-CAM)
- Highlight discriminative regions
- Show top-5 predictions with explanations

**Example Output**:
```
Top Prediction: Golden Retriever (95.6%)
2nd: Labrador Retriever (3.2%)
   → Similar coat color and body shape
   → Check ear shape: Golden has longer, floppier ears
3rd: Flat-Coated Retriever (0.9%)
   → Similar coat but typically darker
```

**Techniques**:
- Grad-CAM for attention maps
- Feature space analysis
- Rule-based breed comparison database

**Effort**: Medium-High (4-6 weeks)

---

## API & Infrastructure

### 8. Rate Limiting & Quotas

**Goal**: Prevent abuse and ensure fair usage

**Features**:
- Per-API-key rate limits (e.g., 100 requests/hour)
- Burst allowance for occasional spikes
- Quota tracking and alerts
- Tiered plans (free, basic, premium)

**Implementation**:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/hour")
async def predict_breed(...):
    ...
```

**Effort**: Low-Medium (1 week)

---

### 9. Async Batch Processing Queue

**Goal**: Handle large batch jobs asynchronously

**Architecture**:
```
Client → API Server → RabbitMQ → Worker Pool → Results DB
                          ↓
                     (Celery Tasks)
```

**Use Case**:
- User uploads 100 images
- API returns job ID immediately
- Workers process in background
- Client polls `/jobs/{job_id}` for status

**Technologies**:
- Celery for task queue
- RabbitMQ or Redis as broker
- PostgreSQL for job status

**Effort**: High (4-6 weeks)

---

### 10. Monitoring & Observability

**Goal**: Track system health and performance

**Metrics**:
- Request rate, latency (p50, p95, p99)
- Error rate (4xx, 5xx)
- Model predictions distribution
- GPU utilization
- Cache hit rate

**Stack**:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Loki**: Log aggregation
- **Alertmanager**: Alert routing

**Dashboards**:
1. **API Performance**: Request rate, latency, errors
2. **Model Performance**: Prediction distribution, confidence scores
3. **Infrastructure**: CPU, GPU, memory usage
4. **Business**: Unique users, popular breeds, trends

**Effort**: Medium-High (3-4 weeks)

---

### 11. CI/CD Pipeline

**Goal**: Automate testing, building, and deployment

**Pipeline Stages**:
1. **Lint**: `flake8`, `black`, `mypy`
2. **Test**: Run test suite with coverage
3. **Build**: Docker image creation
4. **Deploy**: Push to staging, run smoke tests, deploy to production

**Tools**:
- GitHub Actions or GitLab CI
- Docker for containerization
- Kubernetes for orchestration

**Example Workflow**:
```yaml
name: CI/CD
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest --cov=.
  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: kubectl apply -f k8s/
```

**Effort**: Medium (2-3 weeks)

---

## User Experience

### 12. Mobile Application

**Goal**: Native mobile app for iOS and Android

**Framework Options**:
1. **React Native**: JavaScript, cross-platform
2. **Flutter**: Dart, excellent performance
3. **Swift/Kotlin**: Native, best UX but 2× effort

**Features**:
- Camera integration
- Gallery upload
- Real-time prediction
- Breed information display
- History of past predictions
- Favorites/bookmarks

**Recommendation**: Flutter for best performance and single codebase

**Effort**: Very High (12-16 weeks for full app)

---

### 13. Web Application

**Goal**: Browser-based interface for desktop users

**Stack**:
- **Frontend**: React or Vue.js
- **Image Upload**: Drag-and-drop + camera
- **Results Display**: Breed cards with info
- **Responsive**: Mobile-friendly

**Features**:
- Upload multiple images
- Side-by-side breed comparison
- Breed encyclopedia/search
- Share results on social media

**Effort**: High (6-8 weeks)

---

### 14. Progressive Web App (PWA)

**Goal**: Install web app like native app

**Features**:
- Offline support (cached predictions)
- Add to home screen
- Push notifications
- Background sync

**Benefits**:
- No app store approval needed
- Cross-platform (iOS, Android, desktop)
- Automatic updates

**Effort**: Medium (added to web app) (2-3 weeks)

---

## Research Directions

### 15. Few-Shot Learning for Rare Breeds

**Goal**: Recognize rare breeds with few training examples

**Approach**:
- Prototypical networks
- Siamese networks
- Meta-learning (MAML)

**Use Case**:
- Add new breed with only 10-20 images
- Recognize exotic/rare breeds

**Effort**: Very High (research project, 12+ weeks)

---

### 16. Mixed-Breed Classification

**Goal**: Identify mixed-breed dogs and estimate breed composition

**Example Output**:
```
Mixed Breed Detected
Estimated Composition:
- 60% Labrador Retriever
- 30% German Shepherd  
- 10% Border Collie
```

**Approach**:
- Multi-label classification
- Breed embedding space
- Genetic likelihood estimation

**Challenges**:
- Ground truth is difficult (requires DNA test)
- Infinite combinations
- Visual features may not match genetics

**Effort**: Very High (research project, 16+ weeks)

---

### 17. Active Learning for Data Efficiency

**Goal**: Intelligently select most informative samples for labeling

**Process**:
1. Train initial model
2. Identify uncertain predictions
3. Request human labels for those samples
4. Retrain with new labels
5. Repeat

**Benefits**:
- Reduce labeling effort by 50-70%
- Improve model on edge cases
- Continuous improvement

**Effort**: High (6-8 weeks)

---

## Technical Debt

### 18. Code Refactoring

**Areas to Improve**:

#### API Server
- Split `api_server.py` into modules
  - `routes/` for endpoints
  - `services/` for business logic
  - `models/` for Pydantic schemas
  - `utils/` for helpers

#### Configuration Management
- Move hardcoded values to `config.yaml`
- Environment-specific configs (dev, staging, prod)
- Secrets management

#### Error Handling
- Custom exception classes
- Consistent error response format
- Detailed logging

**Effort**: Medium (2-3 weeks)

---

### 19. Documentation

**Additions Needed**:
- **API Documentation**: Expand with more examples
- **Deployment Guide**: Step-by-step cloud deployment
- **Development Guide**: Contributing guidelines, code style
- **Troubleshooting**: Common issues and solutions
- **Performance Tuning**: GPU optimization tips

**Effort**: Low-Medium (1-2 weeks)

---

### 20. Model Versioning

**Goal**: Track model versions and enable A/B testing

**System**:
- Version each trained model (v1.0, v1.1, etc.)
- Store metadata (accuracy, training date, dataset)
- API endpoint to select model version
- A/B testing framework

**Tools**:
- MLflow for experiment tracking
- DVC for model versioning

**Example**:
```bash
# Load specific model version
GET /predict?model_version=v2.1
```

**Effort**: Medium (2-3 weeks)

---

**Last Updated**: June 2026  
**Next Review**: September 2026

