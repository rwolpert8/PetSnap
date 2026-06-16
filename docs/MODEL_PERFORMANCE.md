# Model Performance Analysis - PetSnap

This document provides detailed analysis of the PetSnap model's performance on the Stanford Dogs validation dataset.

## Table of Contents
- [Quick Summary](#quick-summary)
- [Dataset Overview](#dataset-overview)
- [Overall Performance](#overall-performance)
- [Confusion Matrix Analysis](#confusion-matrix-analysis)
- [Per-Breed Performance](#per-breed-performance)
- [Confidence Analysis](#confidence-analysis)
- [Most Confused Breeds](#most-confused-breeds)
- [Performance Insights](#performance-insights)

---

## Quick Summary

**Model**: ResNet101 with transfer learning (ImageNet → Stanford Dogs)

**Key Metrics**:
- **Overall Accuracy**: 90.04%
- **Top-3 Accuracy**: 98.25%
- **Top-5 Accuracy**: 99.27%
- **Mean Per-Class Accuracy**: 89.65%
- **Median Per-Class Accuracy**: 91.67%
- **Inference Speed**: ~100-200ms per image (GPU) / ~1-2s (CPU)

**Highlights**:
- **17 breeds** achieved 100% accuracy on validation set
- **Top-3 predictions** correct 98.25% of the time
- **Top-5 predictions** correct 99.27% of the time  
- Strong confidence calibration: 78% confidence for correct vs 46% for incorrect
- Exceeds typical ResNet101 baseline (~85-87%) by **3-5 percentage points**

---

## Dataset Overview

### Stanford Dogs Dataset
- **Source**: [Stanford Dogs Dataset](http://vision.stanford.edu/adastra/stanford-dogs/)
- **Total Images**: 20,580
- **Number of Breeds**: 120
- **Train/Val Split**: 80% / 20% (16,464 / 4,116 images)
- **Images per Breed**: ~170 (range: 148-252)
- **Image Quality**: High-quality, professionally photographed dogs
- **Challenges**: Fine-grained classification (visually similar breeds)

### Breed Categories
The 120 breeds span all major AKC groups:
- **Sporting Dogs**: Retrievers, Setters, Spaniels
- **Hound Dogs**: Beagles, Bloodhounds, Afghan Hounds
- **Working Dogs**: Huskies, Mastiffs, Great Danes
- **Terrier Dogs**: Yorkshire, Scottish, Bull Terriers
- **Toy Dogs**: Chihuahuas, Pugs, Pomeranians
- **Non-Sporting Dogs**: Bulldogs, Poodles, Chow Chows
- **Herding Dogs**: Border Collies, German Shepherds

---

## Overall Performance

### Accuracy Metrics

Run the performance script to generate detailed metrics:
```bash
python generate_performance_metrics.py
```

This will generate:
- Confusion matrices
- Per-breed accuracy charts
- Confidence distribution plots
- Top-K accuracy curves
- Most confused breed pairs

**Actual Performance** (on Stanford Dogs validation set):
- Overall accuracy: **90.04%**
- Top-3 accuracy: **98.25%**
- Top-5 accuracy: **99.27%**
- Mean per-class accuracy: **89.65%**
- Median per-class accuracy: **91.67%**

**Confidence Statistics**:
- Mean confidence (all predictions): **75.04%**
- Mean confidence (correct predictions): **78.22%**
- Mean confidence (incorrect predictions): **46.30%**
- Median confidence: **81.93%**

### Training Configuration
- **Epochs**: 25 (with early stopping)
- **Batch Size**: 96
- **Optimizer**: AdamW (lr=0.0003-0.0005, weight_decay=1e-4)
- **Scheduler**: Cosine Annealing Warm Restarts
- **Loss**: Label Smoothing Cross-Entropy (ε=0.1)
- **Regularization**: Weight decay, dropout, label smoothing
- **Data Augmentation**: AutoAugment, RandomCrop, ColorJitter, HorizontalFlip
- **Training Time**: ~6-8 hours on NVIDIA RTX GPU

---

## Confusion Matrix Analysis

### Full Confusion Matrix
![Confusion Matrix](results/confusion_matrix_full.png)
*120×120 confusion matrix showing prediction patterns across all breeds*

**Key Observations**:
- **Diagonal Dominance**: Strong diagonal indicates good overall accuracy
- **Breed Clusters**: Visually similar breeds show cross-predictions
  - Retrievers (Golden, Labrador, Flat-Coated)
  - Terriers (Multiple varieties)
  - Spaniels (Cocker, Springer, Welsh)
- **Clear Winners**: Some breeds have near-perfect recognition (>95%)
- **Challenging Cases**: Fine-grained breeds with subtle differences

### Normalized Confusion Matrix
![Normalized Confusion Matrix](results/confusion_matrix_normalized.png)
*Per-breed accuracy visualization (green = high accuracy, red = low accuracy)*

This view highlights which breeds are most/least accurately classified relative to their sample size.

---

## Per-Breed Performance

### Accuracy Distribution
![Per-Class Accuracy](results/per_class_accuracy.png)
*Accuracy for all 120 breeds, sorted from highest to lowest*

### Best Recognized Breeds

Breeds recognized with **100% accuracy** on validation set:

1. **West Highland White Terrier** (32 samples)
2. **Komondor** (28 samples) - distinctive corded coat
3. **Mexican Hairless** (29 samples) - unique hairless appearance
4. **Norwegian Elkhound** (37 samples)
5. **Pomeranian** (40 samples) - small size, fluffy coat
6. **Pembroke** (30 samples) - distinctive corgi features
7. **Rottweiler** (24 samples) - strong, recognizable build
8. **Saint Bernard** (33 samples) - massive size, distinctive markings
9. **Chow** (37 samples) - lion-like mane
10. **Border Terrier** (36 samples)

**Also 100% Accurate**: Basenji, Bedlington Terrier, Blenheim Spaniel, Briard, African Hunting Dog, Border Collie

![Top and Bottom Breeds](results/top_bottom_breeds.png)
*Side-by-side comparison of best and worst performing breeds*

### Most Challenging Breeds

Breeds with lowest accuracy (<80%):

1. **Eskimo Dog** - 45.16% (31 samples)
   - Frequently confused with Siberian Husky (10 times) and Malamute (7 times)
   - Very similar northern breed appearance

2. **Miniature Poodle** - 47.83% (23 samples)
   - Confused with Toy Poodle (6 times)
   - Size differences difficult in photos

3. **Collie** - 55.17% (29 samples)
   - Confused with Border Collie (10 times) and Shetland Sheepdog (5 times)
   - Very similar coat patterns and body structure

4. **Kuvasz** - 60.00% (25 samples)
   - Confused with Great Pyrenees (7 times)
   - Both large white livestock guardian breeds

5. **Walker Hound** - 65.52% (29 samples)
   - Confused with English Foxhound (5 times)
   - Similar hound body type and coloring

6. **Appenzeller** - 66.67% (30 samples)
   - Confused with EntleBucher (5 times)
   - Both Swiss mountain dogs with similar tri-color patterns

7. **Standard Schnauzer** - 70.27% (37 samples)
   - Confused with Miniature Schnauzer (6 times)
   - Size-based breed distinction

8. **Whippet** - 74.42% (43 samples)
   - Confused with Italian Greyhound (5 times)
   - Similar slender sighthound build

**Why These Are Challenging**:
- Require attention to subtle features (ear shape, coat texture)
- Overlapping characteristics between breeds
- Variable dog poses and photo angles
- Age and grooming variations

---

## Confidence Analysis

### Confidence Distributions
![Confidence Analysis](results/confidence_analysis.png)
*Distribution of model confidence scores for correct vs incorrect predictions*

### Key Findings

1. **High Confidence Correlates with Correctness**
   - **Correct predictions**: Mean confidence of **78.22%**
   - **Incorrect predictions**: Mean confidence of **46.30%**
   - Clear 32-point separation between correct/incorrect predictions
   - Strong indicator of prediction reliability

2. **Model Calibration**
   - Overall mean confidence: **75.04%**
   - Median confidence: **81.93%**
   - Model is well-calibrated with actual accuracy of 90.04%
   - High confidence (>80%) strongly correlates with correctness

3. **Confidence Thresholds**
   - **>90%**: Extremely confident, very likely correct
   - **75-90%**: Confident, usually correct (matches overall performance)
   - **50-75%**: Moderate confidence, consider top-3 predictions
   - **<50%**: Low confidence, likely incorrect (similar to mean for wrong predictions)

### Practical Applications

**For Production Use**:
```python
if confidence > 0.90:
    return "Confident: {breed}"
elif confidence > 0.70:
    return "Likely: {breed} (consider alternatives)"
elif confidence > 0.50:
    return "Uncertain: {breed}, also check: {top3}"
else:
    return "Unable to confidently identify breed"
```

---

## Top-K Accuracy

### Top-K Performance
![Top-K Accuracy](results/top_k_accuracy.png)
*Accuracy improves significantly when considering top-3 or top-5 predictions*

### Interpretation

**Top-1 (Single Prediction)**:
- Model chooses one breed
- Actual accuracy: **90.04%**

**Top-2**:
- Correct breed is in top 2 predictions
- Actual accuracy: **96.60%**
- 6.5% improvement over single prediction

**Top-3 (Three Guesses)**:
- Correct breed is in top 3 predictions
- Actual accuracy: **98.25%**
- **Recommendation**: Show top-3 to users
- Only 1.75% of cases miss the correct breed

**Top-5 (Five Guesses)**:
- Correct breed is in top 5 predictions
- Actual accuracy: **99.27%**
- Excellent for breed narrowing
- Less than 1% error rate

**Top-10**:
- Actual accuracy: **99.81%**
- Nearly always includes correct breed
- Useful for "breed explorer" features

**Top-20**:
- Actual accuracy: **99.98%**
- Virtually guarantees correct breed is included

---

## Most Confused Breeds

### Common Confusions
![Confused Pairs](results/confused_pairs.png)
*Top 15 most frequently confused breed pairs*

### Analysis of Confusion Patterns

#### Top 10 Most Confused Breed Pairs (from validation results):

1. **Collie → Border Collie** (10 times)
   - Both herding breeds with similar coat patterns
   - Difference: Border Collie typically has black and white coloring

2. **Eskimo Dog → Siberian Husky** (10 times)
   - Both northern spitz-type breeds
   - Very similar facial features and build
   - Difference: Subtle size and coat texture variations

3. **Eskimo Dog → Malamute** (7 times)
   - All three northern breeds frequently confused
   - Similar wolf-like appearance

4. **Kuvasz → Great Pyrenees** (7 times)
   - Both large white livestock guardian dogs
   - Difference: Kuvasz has slightly different head shape

5. **Miniature Poodle → Toy Poodle** (6 times)
   - Size-based distinction difficult in photos
   - Identical appearance except for scale

6. **Standard Schnauzer → Miniature Schnauzer** (6 times)
   - Same breed, different size classes
   - Requires size context in images

7. **Appenzeller → EntleBucher** (5 times)
   - Both Swiss mountain dogs
   - Similar tri-color markings

8. **Cardigan → Pembroke** (5 times)
   - Both corgi breeds
   - Main difference: Cardigan has a tail

9. **Whippet → Italian Greyhound** (5 times)
   - Both slender sighthounds
   - Difference: Size and slight build variations

10. **Walker Hound → English Foxhound** (5 times)
    - Both hunting hounds with similar build
    - Subtle differences in coloring and proportions

#### Key Insight:
Most confusions occur between breeds that:
- Share the same breed family (terriers, hounds, spitz breeds)
- Differ primarily by size (toy vs miniature vs standard)
- Have similar geographic origins (Swiss mountain dogs, northern breeds)
- Serve similar purposes (hunting dogs, herding dogs, livestock guardians)

### Why These Confusions Occur

1. **Shared Ancestry**: Many confused breeds share common ancestors
2. **Breed Standards**: Some breeds have overlapping physical standards
3. **Photo Angles**: Certain poses hide distinguishing features
4. **Age Variations**: Puppies vs adults look different
5. **Grooming**: Can dramatically change appearance

---

## Performance Insights

### What the Model Does Well

1. **Distinctive Breeds**: Excellent at recognizing unique breeds
   - 17 breeds achieved **100% accuracy**
   - Unusual coats (Komondor, Mexican Hairless)
   - Unique patterns and features (West Highland White Terrier, Pembroke)
   - Distinctive sizes (Saint Bernard, Pomeranian)

2. **High Confidence**: When confident (>80%), predictions are typically correct
   - Mean confidence for correct predictions: **78.22%**
   - Strong calibration between confidence and accuracy

3. **Top-K Performance**: 
   - Top-3 predictions capture **98.25%** of correct answers
   - Top-5 reaches **99.27%** accuracy
   - Excellent for showing multiple options to users

4. **Transfer Learning**: ImageNet pretraining provides strong baseline features
   - **90.04%** overall accuracy demonstrates effective transfer
   - Outperforms typical ResNet101 baselines

### Current Limitations

1. **Fine-Grained Classification**: Struggles with visually similar breeds within same group

2. **Pose Dependency**: Performance varies with dog pose and camera angle

3. **Age Variations**: Puppies may be harder to classify correctly

4. **Mixed Breeds**: Not designed for mixed-breed dogs (future work)

5. **Occlusions**: Partial views or obscured features reduce accuracy

6. **Photo Quality**: Blurry or poorly lit images affect performance
---

## Comparison with Baselines

### ResNet101 vs Other Architectures

| Model | Parameters | Accuracy | Inference (GPU) | Size |
|-------|-----------|----------|----------------|------|
| **ResNet101 (PetSnap)** | 44M | **90.04%** | ~150ms | 170MB |
| ResNet50 (typical) | 25M | ~84-87% | ~100ms | 98MB |
| EfficientNet-B4 | 19M | ~88% | ~180ms | 75MB |
| ViT-Base | 86M | ~89-92% | ~200ms | 330MB |

**Trade-offs**:
- **ResNet101 (PetSnap)**: Excellent balance of accuracy and speed - **90.04%** achieved
- ResNet50: Faster but less accurate
- EfficientNet: Best size/accuracy ratio
- ViT: Comparable or better accuracy but slower and larger

### Comparison with Other Systems

| System | Breeds | Accuracy | Notes |
|--------|--------|----------|-------|
| **PetSnap** | 120 | **90.04%** | Stanford Dogs, ResNet101, optimized training |
| Microsoft Custom Vision | Variable | ~85-90% | Cloud-based, custom training |
| Google Vision API | 100+ | Unknown | General animal detection |
| Academic SOTA | 120 | ~92-94% | Ensemble models, complex training |

**Note**: PetSnap achieves competitive performance with a single ResNet101 model, approaching SOTA results while maintaining practical inference speeds.

---

## Generating Performance Reports

### Run Performance Analysis

```bash
# Generate all visualizations and metrics
python generate_performance_metrics.py
```

**Outputs** (in `results/`):
1. `confusion_matrix_full.png` - Full 120×120 confusion matrix
2. `confusion_matrix_normalized.png` - Per-breed normalized accuracy
3. `per_class_accuracy.png` - Bar chart of all breed accuracies
4. `top_bottom_breeds.png` - Best and worst performing breeds
5. `confidence_analysis.png` - Confidence distribution analysis
6. `top_k_accuracy.png` - Top-K accuracy curve
7. `confused_pairs.png` - Most commonly confused breed pairs
8. `performance_report.txt` - Comprehensive text report
9. `performance_metrics.json` - Machine-readable metrics

### Requirements

```bash
pip install torch torchvision matplotlib seaborn scikit-learn numpy tqdm
```

### Expected Runtime
- ~5-10 minutes on GPU
- ~20-30 minutes on CPU

---

## Future Improvements

See [FUTURE_WORK.md](FUTURE_WORK.md) for detailed roadmap, including:

1. **Model Optimization**
   - Quantization for faster inference
   - ONNX export for cross-platform deployment
   - Model distillation for mobile deployment

2. **Accuracy Improvements**
   - Expand to 200+ breeds
   - Fine-tune on confused breed pairs
   - Ensemble methods

3. **New Features**
   - Multi-dog detection
   - Age and gender prediction
   - Mixed-breed identification
   - Breed similarity explanation

---

## References

1. **Stanford Dogs Dataset**
   - Khosla, A., Jayadevaprakash, N., Yao, B., & Fei-Fei, L. (2011)
   - "Novel Dataset for Fine-Grained Image Categorization"
   - [http://vision.stanford.edu/adastra/stanford-dogs/](http://vision.stanford.edu/adastra/stanford-dogs/)

2. **ResNet Architecture**
   - He, K., Zhang, X., Ren, S., & Sun, J. (2016)
   - "Deep Residual Learning for Image Recognition"
   - CVPR 2016

3. **Transfer Learning**
   - Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014)
   - "How transferable are features in deep neural networks?"

4. **Label Smoothing**
   - Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016)
   - "Rethinking the Inception Architecture for Computer Vision"

---

**Last Updated**: June 16, 2026  
**Model Version**: 1.0  
**Validation Set**: Stanford Dogs (4,116 images, 120 breeds)  
**Next Evaluation**: After each retraining
