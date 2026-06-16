"""
PetSnap Model Performance Visualization Script

This script generates comprehensive performance visualizations for the PetSnap model:
1. Confusion Matrix (120x120 heatmap)
2. Per-Class Accuracy Bar Chart
3. Confidence Distribution Histogram
4. Top-K Accuracy Plot
5. Most Confused Breed Pairs

Usage:
    python generate_performance_metrics.py

Requirements:
    - Trained model (best_model.pth)
    - Validation dataset (./Images/)
    - Libraries: torch, torchvision, matplotlib, seaborn, sklearn, numpy
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import os
import json

# Configuration
MODEL_PATH = '../models/best_model.pth'
DATASET_PATH = '../Images'
OUTPUT_DIR = '.'
BATCH_SIZE = 96
NUM_CLASSES = 120

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model_and_data():
    """Load trained model and validation dataset."""
    print("Loading model and dataset...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = models.resnet101(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Load dataset (use same transform as during inference)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = ImageFolder(root=DATASET_PATH, transform=transform)
    
    # Split dataset (same as training)
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    _, valset = random_split(full_dataset, [train_size, val_size])
    
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    classes = full_dataset.classes
    
    print(f"✓ Dataset loaded: {len(valset)} validation samples, {len(classes)} classes")
    
    return model, valloader, classes, device


def get_predictions(model, dataloader, device):
    """Get predictions and ground truth labels for entire dataset."""
    print("\nGenerating predictions...")
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Inference"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    print("✓ Predictions generated")
    return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)


def plot_confusion_matrix(labels, predictions, classes):
    """Generate and save confusion matrix visualization."""
    print("\nGenerating confusion matrix...")
    
    cm = confusion_matrix(labels, predictions)
    
    # Full confusion matrix
    plt.figure(figsize=(28, 28))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', square=True,
                xticklabels=classes, yticklabels=classes, cbar_kws={'shrink': 0.8})
    plt.title('Confusion Matrix - All 120 Dog Breeds', fontsize=16, pad=20)
    plt.xlabel('Predicted Breed', fontsize=12)
    plt.ylabel('True Breed', fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/confusion_matrix_full.png")
    
    # Normalized confusion matrix (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(28, 28))
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='RdYlGn', square=True,
                xticklabels=classes, yticklabels=classes, cbar_kws={'shrink': 0.8},
                vmin=0, vmax=1)
    plt.title('Normalized Confusion Matrix - Accuracy per Breed', fontsize=16, pad=20)
    plt.xlabel('Predicted Breed', fontsize=12)
    plt.ylabel('True Breed', fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/confusion_matrix_normalized.png")


def plot_per_class_accuracy(labels, predictions, classes):
    """Generate per-class accuracy bar chart."""
    print("\nGenerating per-class accuracy chart...")
    
    # Calculate per-class accuracy
    accuracies = []
    class_counts = []
    
    for i, class_name in enumerate(classes):
        class_mask = labels == i
        class_correct = (predictions[class_mask] == i).sum()
        class_total = class_mask.sum()
        class_counts.append(class_total)
        
        if class_total > 0:
            accuracy = 100.0 * class_correct / class_total
            accuracies.append(accuracy)
        else:
            accuracies.append(0.0)
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    
    # Plot
    plt.figure(figsize=(20, 12))
    colors = ['green' if acc >= 80 else 'orange' if acc >= 60 else 'red' 
              for acc in sorted_accuracies]
    bars = plt.bar(range(len(sorted_classes)), sorted_accuracies, color=colors, alpha=0.7)
    
    plt.axhline(y=np.mean(accuracies), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean Accuracy: {np.mean(accuracies):.2f}%')
    
    plt.xlabel('Dog Breed', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Breed Classification Accuracy (Validation Set)', fontsize=14, pad=20)
    plt.xticks(range(len(sorted_classes)), sorted_classes, rotation=90, fontsize=7)
    plt.ylim(0, 105)
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/per_class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/per_class_accuracy.png")
    
    # Top 10 and bottom 10
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 10
    ax1.barh(range(10), sorted_accuracies[:10][::-1], color='green', alpha=0.7)
    ax1.set_yticks(range(10))
    ax1.set_yticklabels(sorted_classes[:10][::-1], fontsize=10)
    ax1.set_xlabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Top 10 Best Recognized Breeds', fontsize=12)
    ax1.set_xlim(0, 105)
    ax1.grid(axis='x', alpha=0.3)
    
    # Bottom 10
    ax2.barh(range(10), sorted_accuracies[-10:][::-1], color='red', alpha=0.7)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(sorted_classes[-10:][::-1], fontsize=10)
    ax2.set_xlabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Top 10 Most Challenging Breeds', fontsize=12)
    ax2.set_xlim(0, 105)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/top_bottom_breeds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/top_bottom_breeds.png")
    
    return accuracies, class_counts


def plot_confidence_distribution(labels, predictions, probabilities):
    """Plot confidence score distributions."""
    print("\nGenerating confidence distribution plots...")
    
    # Get confidence scores for correct and incorrect predictions
    correct_mask = labels == predictions
    incorrect_mask = ~correct_mask
    
    correct_confidences = np.max(probabilities[correct_mask], axis=1) * 100
    incorrect_confidences = np.max(probabilities[incorrect_mask], axis=1) * 100
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram of all predictions
    axes[0, 0].hist(correct_confidences, bins=50, alpha=0.7, color='green', 
                    label='Correct', density=True)
    axes[0, 0].hist(incorrect_confidences, bins=50, alpha=0.7, color='red', 
                    label='Incorrect', density=True)
    axes[0, 0].set_xlabel('Confidence (%)', fontsize=10)
    axes[0, 0].set_ylabel('Density', fontsize=10)
    axes[0, 0].set_title('Confidence Distribution: Correct vs Incorrect', fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Box plot
    axes[0, 1].boxplot([correct_confidences, incorrect_confidences],
                        labels=['Correct', 'Incorrect'],
                        patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[0, 1].set_ylabel('Confidence (%)', fontsize=10)
    axes[0, 1].set_title('Confidence Statistics', fontsize=11)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Cumulative distribution
    sorted_correct = np.sort(correct_confidences)
    sorted_incorrect = np.sort(incorrect_confidences)
    axes[1, 0].plot(sorted_correct, np.linspace(0, 100, len(sorted_correct)), 
                    'g-', label='Correct', linewidth=2)
    axes[1, 0].plot(sorted_incorrect, np.linspace(0, 100, len(sorted_incorrect)), 
                    'r-', label='Incorrect', linewidth=2)
    axes[1, 0].set_xlabel('Confidence (%)', fontsize=10)
    axes[1, 0].set_ylabel('Cumulative %', fontsize=10)
    axes[1, 0].set_title('Cumulative Confidence Distribution', fontsize=11)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Confidence vs accuracy
    confidence_bins = np.arange(0, 101, 10)
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(confidence_bins) - 1):
        lower, upper = confidence_bins[i], confidence_bins[i+1]
        max_probs = np.max(probabilities, axis=1) * 100
        mask = (max_probs >= lower) & (max_probs < upper)
        
        if mask.sum() > 0:
            bin_accuracy = (predictions[mask] == labels[mask]).mean() * 100
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    axes[1, 1].bar(confidence_bins[:-1], bin_accuracies, width=8, alpha=0.7, 
                   color='steelblue', edgecolor='black')
    axes[1, 1].plot([0, 100], [0, 100], 'r--', label='Perfect Calibration', linewidth=2)
    axes[1, 1].set_xlabel('Confidence Bin (%)', fontsize=10)
    axes[1, 1].set_ylabel('Actual Accuracy (%)', fontsize=10)
    axes[1, 1].set_title('Model Calibration', fontsize=11)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/confidence_analysis.png")
    
    # Statistics
    print(f"\n  Correct predictions - Mean confidence: {correct_confidences.mean():.2f}%")
    print(f"  Incorrect predictions - Mean confidence: {incorrect_confidences.mean():.2f}%")


def plot_top_k_accuracy(labels, probabilities):
    """Plot Top-K accuracy (1, 3, 5, 10)."""
    print("\nCalculating Top-K accuracy...")
    
    k_values = [1, 2, 3, 5, 10, 20]
    accuracies = []
    
    for k in k_values:
        top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
        top_k_correct = np.array([labels[i] in top_k_preds[i] for i in range(len(labels))])
        accuracy = 100.0 * top_k_correct.mean()
        accuracies.append(accuracy)
        print(f"  Top-{k} Accuracy: {accuracy:.2f}%")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'o-', linewidth=2, markersize=8, color='steelblue')
    plt.xlabel('K (Top-K Predictions)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Top-K Accuracy on Validation Set', fontsize=14)
    plt.grid(alpha=0.3)
    plt.ylim(0, 105)
    
    for k, acc in zip(k_values, accuracies):
        plt.text(k, acc + 1, f'{acc:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/top_k_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/top_k_accuracy.png")


def find_most_confused_pairs(labels, predictions, classes):
    """Find the most commonly confused breed pairs."""
    print("\nFinding most confused breed pairs...")
    
    cm = confusion_matrix(labels, predictions)
    
    # Extract off-diagonal elements (misclassifications)
    confusion_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((classes[i], classes[j], cm[i, j]))
    
    # Sort by frequency
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Top 20 confused pairs
    print("\n  Top 20 Most Confused Breed Pairs:")
    for i, (true_breed, pred_breed, count) in enumerate(confusion_pairs[:20], 1):
        print(f"  {i:2d}. {true_breed:30s} → {pred_breed:30s} ({count} times)")
    
    # Visualize top 15
    top_pairs = confusion_pairs[:15]
    pair_labels = [f"{true_breed[:15]}\n→{pred[:15]}" 
                   for true_breed, pred, _ in top_pairs]
    counts = [count for _, _, count in top_pairs]
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(counts)))
    bars = plt.barh(range(len(pair_labels)), counts, color=colors)
    plt.yticks(range(len(pair_labels)), pair_labels, fontsize=9)
    plt.xlabel('Number of Misclassifications', fontsize=11)
    plt.title('Top 15 Most Confused Breed Pairs', fontsize=13)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confused_pairs.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR}/confused_pairs.png")
    
    return confusion_pairs[:20]


def generate_metrics_report(labels, predictions, probabilities, classes, accuracies, 
                           class_counts, confused_pairs):
    """Generate comprehensive text report with all metrics."""
    print("\nGenerating metrics report...")
    
    # Overall metrics
    overall_accuracy = 100.0 * (predictions == labels).mean()
    
    # Top-1, Top-3, Top-5
    top_k_preds = np.argsort(probabilities, axis=1)
    top3_accuracy = 100.0 * np.array([labels[i] in top_k_preds[i, -3:] 
                                      for i in range(len(labels))]).mean()
    top5_accuracy = 100.0 * np.array([labels[i] in top_k_preds[i, -5:] 
                                      for i in range(len(labels))]).mean()
    
    # Classification report
    report = classification_report(labels, predictions, target_names=classes, 
                                   digits=4, zero_division=0)
    
    # Write report
    report_path = f'{OUTPUT_DIR}/performance_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PetSnap Model Performance Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Validation Samples: {len(labels)}\n")
        f.write(f"Number of Classes: {len(classes)}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
        f.write(f"Top-3 Accuracy: {top3_accuracy:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5_accuracy:.2f}%\n")
        f.write(f"Mean Per-Class Accuracy: {np.mean(accuracies):.2f}%\n")
        f.write(f"Median Per-Class Accuracy: {np.median(accuracies):.2f}%\n\n")
        
        f.write("CONFIDENCE STATISTICS\n")
        f.write("-" * 80 + "\n")
        max_probs = np.max(probabilities, axis=1) * 100
        correct_mask = predictions == labels
        f.write(f"Mean Confidence (All): {max_probs.mean():.2f}%\n")
        f.write(f"Mean Confidence (Correct): {max_probs[correct_mask].mean():.2f}%\n")
        f.write(f"Mean Confidence (Incorrect): {max_probs[~correct_mask].mean():.2f}%\n")
        f.write(f"Median Confidence: {np.median(max_probs):.2f}%\n\n")
        
        f.write("TOP 10 BEST RECOGNIZED BREEDS\n")
        f.write("-" * 80 + "\n")
        sorted_indices = np.argsort(accuracies)[::-1]
        for i, idx in enumerate(sorted_indices[:10], 1):
            f.write(f"{i:2d}. {classes[idx]:35s} - {accuracies[idx]:6.2f}% "
                   f"({class_counts[idx]} samples)\n")
        
        f.write("\nTOP 10 MOST CHALLENGING BREEDS\n")
        f.write("-" * 80 + "\n")
        for i, idx in enumerate(sorted_indices[-10:][::-1], 1):
            f.write(f"{i:2d}. {classes[idx]:35s} - {accuracies[idx]:6.2f}% "
                   f"({class_counts[idx]} samples)\n")
        
        f.write("\nTOP 20 MOST CONFUSED BREED PAIRS\n")
        f.write("-" * 80 + "\n")
        for i, (true_breed, pred_breed, count) in enumerate(confused_pairs, 1):
            f.write(f"{i:2d}. {true_breed:30s} → {pred_breed:30s} ({count} times)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    
    print(f"✓ Saved: {report_path}")
    
    # Also save as JSON
    metrics_json = {
        "overall_accuracy": float(overall_accuracy),
        "top3_accuracy": float(top3_accuracy),
        "top5_accuracy": float(top5_accuracy),
        "mean_per_class_accuracy": float(np.mean(accuracies)),
        "median_per_class_accuracy": float(np.median(accuracies)),
        "mean_confidence": float(max_probs.mean()),
        "per_class_accuracies": {classes[i]: float(accuracies[i]) 
                                 for i in range(len(classes))},
        "top_breeds": [{"breed": classes[idx], "accuracy": float(accuracies[idx])}
                       for idx in sorted_indices[:10]],
        "challenging_breeds": [{"breed": classes[idx], "accuracy": float(accuracies[idx])}
                              for idx in sorted_indices[-10:][::-1]],
        "confused_pairs": [{"true_breed": t, "predicted_breed": p, "count": int(c)}
                          for t, p, c in confused_pairs]
    }
    
    json_path = f'{OUTPUT_DIR}/performance_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"✓ Saved: {json_path}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("PetSnap Model Performance Analysis")
    print("=" * 80)
    
    # Load model and data
    model, valloader, classes, device = load_model_and_data()
    
    # Get predictions
    labels, predictions, probabilities = get_predictions(model, valloader, device)
    
    # Generate visualizations
    plot_confusion_matrix(labels, predictions, classes)
    accuracies, class_counts = plot_per_class_accuracy(labels, predictions, classes)
    plot_confidence_distribution(labels, predictions, probabilities)
    plot_top_k_accuracy(labels, probabilities)
    confused_pairs = find_most_confused_pairs(labels, predictions, classes)
    
    # Generate report
    generate_metrics_report(labels, predictions, probabilities, classes, 
                           accuracies, class_counts, confused_pairs)
    
    print("\n" + "=" * 80)
    print(f"✓ All performance visualizations saved to: {OUTPUT_DIR}/")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - confusion_matrix_full.png")
    print("  - confusion_matrix_normalized.png")
    print("  - per_class_accuracy.png")
    print("  - top_bottom_breeds.png")
    print("  - confidence_analysis.png")
    print("  - top_k_accuracy.png")
    print("  - confused_pairs.png")
    print("  - performance_report.txt")
    print("  - performance_metrics.json")


if __name__ == "__main__":
    main()
