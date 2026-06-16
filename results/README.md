# Results

This folder contains training results, performance visualizations, and the metrics generation script.

## Contents

### Performance Metrics Script
- **`generate_performance_metrics.py`** - Script to generate comprehensive performance visualizations

### Generated Visualizations
When you run the performance metrics script, the following outputs will be created in this folder:

1. **`confusion_matrix_full.png`** - Full 120×120 confusion matrix
2. **`confusion_matrix_normalized.png`** - Per-breed normalized accuracy
3. **`per_class_accuracy.png`** - Bar chart of all breed accuracies  
4. **`top_bottom_breeds.png`** - Best and worst performing breeds
5. **`confidence_analysis.png`** - Confidence distribution analysis
6. **`top_k_accuracy.png`** - Top-K accuracy curve
7. **`confused_pairs.png`** - Most commonly confused breed pairs
8. **`performance_report.txt`** - Comprehensive text report
9. **`performance_metrics.json`** - Machine-readable metrics

## Usage

To generate the performance visualizations:

```bash
cd results
python generate_performance_metrics.py
```

This will load the trained model from `../models/best_model.pth` and the dataset from `../Images/`, then generate all performance metrics and save them to this folder.

## Expected Runtime

- With GPU: ~5-10 minutes
- With CPU: ~15-30 minutes

## Git Ignore

This folder and its contents (except this README and the script) are excluded from git version control to avoid committing large image files and generated outputs.

## Folder Structure (After Running)

```
results/
├── generate_performance_metrics.py  # Metrics generation script
├── README.md                        # This file
├── confusion_matrix_full.png        # Generated visualizations
├── confusion_matrix_normalized.png
├── per_class_accuracy.png
├── top_bottom_breeds.png
├── confidence_analysis.png
├── top_k_accuracy.png
├── confused_pairs.png
├── performance_report.txt
└── performance_metrics.json
```
