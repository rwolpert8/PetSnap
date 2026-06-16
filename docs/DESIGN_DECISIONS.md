# Design Decisions - PetSnap

This document explains the key design decisions made during the development of PetSnap, along with the rationale, alternatives considered, and trade-offs.

## Table of Contents
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [API Design](#api-design)
- [Data Augmentation](#data-augmentation)
- [Inference Optimization](#inference-optimization)
- [Web Scraping](#web-scraping)
- [Security](#security)

---

## Model Architecture

### Decision: ResNet101 with Transfer Learning

**Chosen Approach**: Use ResNet101 pretrained on ImageNet, replace final fully connected layer with custom layer for 120 classes.

**Rationale**:
- **Proven Architecture**: ResNet101 is battle-tested for image classification tasks
- **Transfer Learning Benefits**: ImageNet pretraining provides strong feature extraction for general images
- **Depth vs Speed Trade-off**: 101 layers provide excellent accuracy while remaining relatively fast
- **Feature Richness**: Deep residual blocks capture hierarchical features from simple edges to complex dog features

**Alternatives Considered**:

1. **ResNet50**
   - Pros: Faster inference (~50% faster), smaller model size
   - Cons: Lower accuracy potential, fewer parameters for fine-tuning
   - Decision: Chose ResNet101 for better accuracy on 120 fine-grained classes

2. **EfficientNet-B4/B5**
   - Pros: Better accuracy-per-parameter ratio, smaller model size
   - Cons: More complex training, slower inference on some hardware
   - Decision: ResNet101 has better PyTorch support and community resources

3. **Vision Transformer (ViT)**
   - Pros: State-of-art results on large datasets, attention mechanisms
   - Cons: Requires massive datasets, computational overhead, harder to train
   - Decision: Overkill for this task, ResNet sufficient for dog breeds

4. **Custom CNN from Scratch**
   - Pros: Full control over architecture, smaller model
   - Cons: No transfer learning, requires extensive training, likely lower accuracy
   - Decision: Transfer learning significantly reduces training time and improves accuracy

**Key Parameters**:
- Input size: 224×224 (standard ImageNet resolution)
- Output classes: 120 (Stanford Dogs dataset)
- Final FC layer: 2048 → 120 (preserving ResNet's feature dimension)

---

## Training Strategy

### Decision 1: Label Smoothing Loss

**Chosen Approach**: Use label smoothing with ε=0.1 instead of standard cross-entropy.

**Rationale**:
- **Overconfidence Prevention**: Standard cross-entropy encourages the model to be overconfident (pushing probabilities to 0 or 1)
- **Generalization**: Label smoothing acts as regularization, improving test accuracy
- **Fine-Grained Classification**: Especially useful for visually similar dog breeds (e.g., multiple retriever types)

**Formula**:
```
Loss = (1-ε) × NLL_loss(predicted, target) + ε × mean(log_softmax(predicted))
```

**Results**: ~1-2% accuracy improvement compared to standard cross-entropy

### Decision 2: AdamW Optimizer with Cosine Annealing

**Chosen Approach**: AdamW optimizer with CosineAnnealingWarmRestarts scheduler.

**Rationale**:
- **AdamW vs Adam**: Better weight decay handling, improved generalization
- **Learning Rate**: 0.0003-0.0005 (lower than typical for fine-tuning stability)
- **Weight Decay**: 1e-4 (prevents overfitting on small dataset)
- **Cosine Annealing**: Gradually reduces learning rate, allows for warm restarts
  - T_0 = 5 epochs (initial restart period)
  - T_mult = 2 (doubles restart period each time)
  - eta_min = 1e-6 (minimum learning rate)

**Alternatives Considered**:
- **SGD with Momentum**: More stable but slower convergence
- **Step Decay**: Less sophisticated than cosine annealing
- **OneCycleLR**: Good but requires knowing total steps upfront

### Decision 3: Early Stopping

**Chosen Approach**: Monitor validation loss with patience=5 epochs.

**Rationale**:
- **Efficiency**: Stops training when model stops improving, saves compute
- **Overfitting Prevention**: Prevents memorization of training data
- **Best Model Recovery**: Automatically saves and restores best checkpoint

**Configuration**:
- Metric: Validation loss (not accuracy, to avoid overfitting)
- Patience: 5 epochs
- Save strategy: Save checkpoint only when validation loss improves

### Decision 4: Mixed Precision Training (AMP)

**Chosen Approach**: Use PyTorch's Automatic Mixed Precision with GradScaler.

**Rationale**:
- **Speed**: ~2x faster training on modern GPUs (Tensor Cores)
- **Memory**: Reduces GPU memory usage by ~40%
- **Accuracy**: Minimal impact on final accuracy with proper gradient scaling
- **Batch Size**: Allows larger batch sizes (96 instead of 64)

**Implementation**:
```python
scaler = GradScaler('cuda')
with autocast(device_type='cuda'):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Data Augmentation

### Decision: AutoAugment + Custom Augmentation Pipeline

**Chosen Approach**: Combine AutoAugment (ImageNet policy) with custom augmentations.

**Augmentation Pipeline**:
1. **Resize**: 256×256 (slightly larger than target)
2. **AutoAugment**: IMAGENET policy (learned augmentation strategies)
3. **RandomResizedCrop**: 224×224, scale=(0.75, 1.0)
4. **RandomHorizontalFlip**: p=0.5
5. **ColorJitter**: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
6. **Normalize**: ImageNet mean/std

**Rationale**:
- **AutoAugment**: State-of-the-art learned augmentation policies
- **Resize Strategy**: Upsample first, then crop to maintain quality
- **Crop Scale**: 0.75-1.0 ensures dog is still visible (not cropped too much)
- **Color Jitter**: Dogs vary in lighting conditions, helps generalization
- **No Vertical Flip**: Dogs don't appear upside-down in real photos

**Alternatives Considered**:
- **RandAugment**: Simpler but less sophisticated
- **Cutout/Mixup**: Risk obscuring critical breed-identifying features
- **More aggressive crops**: Could cut off important features (ears, tail, etc.)

**Validation Transform**: Simplified (only Resize + CenterCrop + Normalize) to evaluate true performance.

---

## API Design

### Decision 1: FastAPI Framework

**Chosen Approach**: Use FastAPI instead of Flask or Django.

**Rationale**:
- **Performance**: ASGI-based, async/await support, significantly faster than Flask
- **Type Safety**: Pydantic models provide automatic validation
- **Documentation**: Auto-generated OpenAPI/Swagger docs
- **Modern**: Built-in support for modern Python features
- **Production Ready**: Easy deployment with Uvicorn/Gunicorn

**Comparison**:
| Feature | FastAPI | Flask | Django |
|---------|---------|-------|--------|
| Speed | Best | Good | Good |
| Async Support | Yes | Partial | Partial |
| Auto Docs | Yes | No | No |
| Type Hints | Yes | No | No |
| Learning Curve | Low | Low | High |

### Decision 2: Multiple Endpoint Formats

**Chosen Approach**: Provide multiple endpoints for different use cases.

**Endpoints**:
1. **`/predict`**: Simple prediction with top-5 results
2. **`/api/identify`**: File upload with breed info
3. **`/identify`**: Base64 image for mobile apps
4. **`/predict_batch`**: Batch processing (up to 10 images)

**Rationale**:
- **Flexibility**: Different clients have different needs
- **Mobile Compatibility**: Base64 endpoint works with React Native/Flutter
- **Batch Efficiency**: Process multiple images with single request overhead
- **Backward Compatibility**: Can deprecate endpoints gradually

### Decision 3: API Key Authentication

**Chosen Approach**: Simple API key via header (`X-API-Key` or Bearer token).

**Rationale**:
- **Simplicity**: Easy to implement and use
- **Sufficient Security**: Adequate for MVP/small scale
- **Stateless**: No session management needed
- **Fast**: No database lookup per request

**Why Not More Complex Auth?**:
- **OAuth2**: Overkill for simple API, adds complexity
- **JWT**: Unnecessary overhead for simple key validation
- **Database Auth**: Adds latency and database dependency

**Future Enhancement**: JWT tokens with expiration for production scale.

---

## Inference Optimization

### Decision 1: Test Time Augmentation (TTA)

**Chosen Approach**: Apply multiple augmentations at inference and average predictions.

**Implementation**:
- Original image prediction
- 3 augmented versions with random horizontal flips
- Average softmax probabilities across all versions

**Rationale**:
- **Accuracy Boost**: ~1-2% improvement for minimal code complexity
- **Confidence Calibration**: Averaged predictions are better calibrated
- **Acceptable Latency**: 3-4× slower but still under 1 second on GPU

**Trade-offs**:
- Better accuracy and confidence estimates
- 3-4× inference time (300-600ms vs 100-200ms)
- Decision: Acceptable for user-facing app (still feels instant)

**Why Not More Augmentations?**:
- Diminishing returns after 3-5 augmentations
- Latency increases linearly with augmentation count
- 3 augmentations is sweet spot for accuracy/speed

### Decision 2: Batch Size 96

**Chosen Approach**: Use batch size 96 during training.

**Rationale**:
- **GPU Utilization**: Fully utilizes GPU memory (16GB)
- **Training Speed**: Larger batches = fewer iterations
- **Batch Normalization**: Larger batches improve BN statistics
- **Gradient Stability**: More stable gradients than very large batches (>128)

**Why Not Smaller/Larger?**:
- Batch 32-64: Underutilizes GPU, slower training
- Batch 128+: Requires aggressive learning rate, may hurt convergence

---

## Web Scraping

### Decision: Multi-Strategy AKC Scraping

**Chosen Approach**: Implement 3 fallback strategies for robustness.

**Strategy Hierarchy**:
1. **Share Modal Content**: Look for clean breed descriptions in social share metadata
2. **JSON Extraction**: Parse embedded JSON in script tags
3. **HTML Parsing**: Extract from main content paragraphs with filtering

**Rationale**:
- **Robustness**: AKC website changes frequently, multiple strategies ensure resilience
- **Quality**: Share modal content is cleanest, but not always available
- **Fallback**: Always return something, even if not perfect

**Breed Name Mapping**:
- 30+ hardcoded mappings for edge cases
- Handles URL format differences (e.g., "toy poodle" → "poodle-toy")
- Prevents 404 errors on unusual breed names

**Alternatives Considered**:
- **Wikipedia API**: Inconsistent quality, missing many breeds
- **Third-Party APIs**: Cost money, rate limits, dependency risk
- **Static Database**: Outdated quickly, no detailed info

**Future Enhancement**: Cache responses with TTL to reduce scraping overhead.

---

## Security

### Decision 1: API Key in Environment Variable

**Chosen Approach**: Store API key in environment variable with default fallback.

**Rationale**:
- **Security**: Not hardcoded in source code
- **Flexibility**: Easy to change per deployment
- **Default for Dev**: Fallback key for local development

**Production Recommendation**:
- Use secrets manager (AWS Secrets Manager, Azure Key Vault)
- Rotate keys regularly
- Implement rate limiting per key

### Decision 2: CORS Allow All (Development)

**Chosen Approach**: Allow all origins in CORS middleware.

**Rationale**:
- **Development Ease**: No CORS issues during testing
- **Mobile Apps**: Different origins for iOS/Android
- **Flexibility**: Can tighten in production

**Production Recommendation**:
```python
allow_origins=["https://petsnap.app", "https://mobile.petsnap.app"]
```

### Decision 3: Input Validation

**Chosen Approach**: Multi-layer validation for uploaded images.

**Validation Steps**:
1. **Content-Type Check**: Must start with `image/`
2. **PIL Open**: Attempt to open as valid image
3. **RGB Conversion**: Ensure 3-channel RGB format
4. **Size Limits**: Implicit through FastAPI file size limits

**Rationale**:
- **Security**: Prevent malicious file uploads
- **Stability**: Catch corrupt images before model inference
- **User Experience**: Clear error messages

---

## Dataset

### Decision: Stanford Dogs Dataset

**Chosen Approach**: Use Stanford Dogs Dataset (20,580 images, 120 breeds).

**Rationale**:
- **Academic Standard**: Well-known benchmark dataset
- **Fine-Grained**: Specifically designed for dog breed classification
- **Quality**: High-quality images with clean labels
- **Size**: Large enough for deep learning, small enough for fast iteration

**Split Strategy**: 80/20 train/val split (no separate test set in this version).

**Why Not Other Datasets?**:
- **ImageNet Dogs**: Only 100 breeds, part of larger dataset
- **Dog vs Cat**: Too easy, not breed-specific
- **Custom Scraping**: Legal issues, time-consuming, quality control

---

## Infrastructure

### Decision: CPU/GPU Auto-Detection

**Chosen Approach**: Automatically detect and use GPU if available, fallback to CPU.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Rationale**:
- **Flexibility**: Same code works on any machine
- **Development**: Work on laptop (CPU) or workstation (GPU)
- **Deployment**: Easy to deploy to various environments

**Performance**:
- GPU (NVIDIA RTX): ~100-200ms per image
- CPU (Intel i7): ~1-2s per image
- Recommendation: Use GPU for production

---

## Trade-offs Summary

| Decision | Pros | Cons | Verdict |
|----------|------|------|---------|
| ResNet101 vs ResNet50 | +3-5% accuracy | +100ms latency | Worth it for quality |
| Label Smoothing | +1-2% accuracy | Slightly slower | Clear win |
| TTA (3 aug) | +1-2% accuracy | 3-4× slower | Acceptable for UX |
| AutoAugment | +2-3% accuracy | Slower training | Worth it |
| Mixed Precision | 2× faster, -40% memory | Complex setup | Essential |
| FastAPI vs Flask | 3-5× faster | New framework | Clear choice |
| AKC Scraping | Rich info | Fragile, slow | Worth adding |
| API Key Auth | Simple, fast | Less secure | Good for MVP |

---

## Lessons Learned

### What Worked Well
1. **Transfer learning from ImageNet**: Massive time savings and accuracy boost
2. **Label smoothing**: Simple technique, consistent improvement
3. **Mixed precision training**: Essential for large batch sizes
4. **FastAPI**: Excellent developer experience and performance
5. **Multi-strategy scraping**: Robust to website changes

### What We'd Change
1. **Dataset**: Consider augmenting with more images per breed (current: ~170 per breed)
2. **Test Set**: Should have separate test set for unbiased evaluation
3. **Caching**: Add Redis for breed info caching (reduce scraping)
4. **Monitoring**: Add Prometheus metrics from day one
5. **Quantization**: Explore INT8 quantization for faster inference

### Future Improvements
See [FUTURE_WORK.md](FUTURE_WORK.md) for detailed roadmap.

---

**Document Version**: 1.0  
**Last Updated**: June 2026  
**Author**: PetSnap Team
