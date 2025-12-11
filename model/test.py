"""
test.py - Model Evaluation and Metrics Generation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    top_k_accuracy_score
)
from typing import Dict, List, Tuple, Optional
import json
from utils.config import Config
from tqdm import tqdm
import pandas as pd
from datetime import datetime


class Tester:
    """Comprehensive model testing and evaluation class."""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        confidence_threshold: float = 0.5  # Nouveau paramètre
    ):
        """
        Initialize the tester.

        Args:
            model: Trained PyTorch model
            config: Configuration object
            confidence_threshold: Seuil minimum de confiance (0-1)
        """
        self.config = config
        self.model = model.to(self.config.device)
        self.confidence_threshold = confidence_threshold  # Ajouter ceci

        # Reste du code...
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.config.working_directory / "test" / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
        self.all_confidences = []  # Nouveau: stocker les confiances
        self.uncertain_mask = []   # Nouveau: masque pour les prédictions incertaines
        self.class_names = []

    def evaluate(
        self,
        test_loader: DataLoader,
        save_results: bool = True,
        use_confidence_threshold: bool = True  # Nouveau paramètre
    ) -> Dict:
        """
        Run evaluation on test dataset.

        Args:
            test_loader: DataLoader for test data
            save_results: Whether to save results to disk
            use_confidence_threshold: Si True, applique le seuil de confiance

        Returns:
            Dictionary containing all metrics
        """
        print(f"Starting evaluation on {len(test_loader.dataset)} samples...")
        if use_confidence_threshold:
            print(f"Confidence threshold: {self.confidence_threshold}")

        self.model.eval()
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
        self.all_confidences = []
        self.uncertain_mask = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing'):
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)

                # Forward pass
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)

                # Récupérer la confiance maximale et la prédiction
                max_probs, predicted = torch.max(probabilities, 1)

                # Appliquer le seuil de confiance
                if use_confidence_threshold:
                    uncertain = max_probs < self.confidence_threshold
                    self.uncertain_mask.extend(uncertain.cpu().numpy())
                else:
                    uncertain = torch.zeros_like(predicted, dtype=torch.bool)
                    self.uncertain_mask.extend(uncertain.cpu().numpy())

                # Store results
                self.all_predictions.extend(predicted.cpu().numpy())
                self.all_labels.extend(labels.cpu().numpy())
                self.all_probabilities.extend(probabilities.cpu().numpy())
                self.all_confidences.extend(max_probs.cpu().numpy())

        # Convert to numpy arrays
        self.all_predictions = np.array(self.all_predictions)
        self.all_labels = np.array(self.all_labels)
        self.all_probabilities = np.array(self.all_probabilities)
        self.all_confidences = np.array(self.all_confidences)
        self.uncertain_mask = np.array(self.uncertain_mask)

        # Get class names
        self.class_names = self.config.class_mapping.idx_to_name

        # Calculate all metrics
        metrics = self._calculate_metrics(use_confidence_threshold)

        if save_results:
            self._save_results(metrics, use_confidence_threshold)

        return metrics

    def _calculate_metrics(self, use_confidence_threshold: bool = True) -> Dict:
        """Calculate all evaluation metrics."""
        metrics = {}

        # Créer des versions filtrées des prédictions
        if use_confidence_threshold:
            # Masquer les prédictions incertaines
            confident_mask = ~self.uncertain_mask
            confident_predictions = self.all_predictions[confident_mask]
            confident_labels = self.all_labels[confident_mask]

            n_total = len(self.all_predictions)
            n_uncertain = self.uncertain_mask.sum()
            n_confident = confident_mask.sum()

            metrics['confidence_stats'] = {
                'total_samples': int(n_total),
                'uncertain_samples': int(n_uncertain),
                'confident_samples': int(n_confident),
                'uncertain_ratio': float(n_uncertain / n_total),
                'mean_confidence': float(self.all_confidences.mean()),
                'median_confidence': float(np.median(self.all_confidences))
            }

            print(f"\n📊 Confidence Statistics:")
            print(f"  Samples incertains ('not_sure'): {n_uncertain}/{n_total} ({100 * n_uncertain / n_total:.2f}%)")
            print(f"  Confiance moyenne: {self.all_confidences.mean():.4f}")
        else:
            confident_predictions = self.all_predictions
            confident_labels = self.all_labels
            confident_mask = np.ones(len(self.all_predictions), dtype=bool)

        # Calculate metrics only on confident predictions
        if len(confident_predictions) > 0:
            metrics['overall_accuracy'] = accuracy_score(
                confident_labels,
                confident_predictions
            )

            # Top-k accuracy (seulement sur prédictions confiantes)
            if len(confident_predictions) > 0:
                metrics['top_3_accuracy'] = top_k_accuracy_score(
                    confident_labels,
                    self.all_probabilities[confident_mask],
                    k=min(3, len(self.class_names))
                )
                metrics['top_5_accuracy'] = top_k_accuracy_score(
                    confident_labels,
                    self.all_probabilities[confident_mask],
                    k=min(5, len(self.class_names))
                )

            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                confident_labels,
                confident_predictions,
                average=None,
                zero_division=0
            )

            metrics['per_class'] = {
                self.class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(self.class_names))
            }

            # Macro and weighted averages
            for avg_type in ['macro', 'weighted']:
                p, r, f, _ = precision_recall_fscore_support(
                    confident_labels,
                    confident_predictions,
                    average=avg_type,
                    zero_division=0
                )
                metrics[f'{avg_type}_precision'] = float(p)
                metrics[f'{avg_type}_recall'] = float(r)
                metrics[f'{avg_type}_f1'] = float(f)

            # Confusion matrix (seulement sur prédictions confiantes)
            metrics['confusion_matrix'] = confusion_matrix(
                confident_labels,
                confident_predictions
            ).tolist()
        else:
            print("⚠️ Aucune prédiction confiante!")
            metrics['overall_accuracy'] = 0.0

        return metrics

    def plot_confusion_matrix(
        self,
        normalize: bool = False,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.all_labels, self.all_predictions)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'

        n = len(self.class_names)

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot= n<10 if normalize else n<15, # Normalised takes a bit more space
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        filename = f'confusion_matrix{"_normalized" if normalize else ""}.png'
        plt.savefig(self.run_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")

    def plot_per_class_metrics(self, figsize: Tuple[int, int] = (14, 6)):
        """Plot per-class precision, recall, and F1 scores."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.all_labels,
            self.all_predictions,
            average=None,
            zero_division=0
        )

        x = np.arange(len(self.class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1 Score', alpha=0.8)

        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(self.run_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: per_class_metrics.png")

    def plot_roc_curves(self, figsize: Tuple[int, int] = (10, 8)):
        """Plot ROC curves for each class (one-vs-rest)."""
        n_classes = len(self.class_names)

        # Binarize labels for one-vs-rest
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(self.all_labels, classes=range(n_classes))

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

        for i, (color, class_name) in enumerate(zip(colors, self.class_names)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], self.all_probabilities[:, i])
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr, tpr,
                color=color,
                lw=2,
                label=f'{class_name} (AUC = {roc_auc:.2f})',
                alpha=0.7
            )

        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves (One-vs-Rest)')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.run_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: roc_curves.png")

    def plot_top_k_accuracy(self, max_k: int = 10):
        """Plot top-k accuracy for different k values."""
        k_values = range(1, min(max_k + 1, len(self.class_names) + 1))
        accuracies = []

        for k in k_values:
            acc = top_k_accuracy_score(
                self.all_labels,
                self.all_probabilities,
                k=k
            )
            accuracies.append(acc)

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8)
        plt.xlabel('k')
        plt.ylabel('Top-k Accuracy')
        plt.title('Top-k Accuracy vs k')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)
        plt.ylim([0, 1.05])

        for k, acc in zip(k_values, accuracies):
            plt.text(k, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.run_dir / 'top_k_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: top_k_accuracy.png")

    def plot_misclassification_analysis(self, top_n: int = 10):
        """Analyze and visualize most common misclassifications."""
        cm = confusion_matrix(self.all_labels, self.all_predictions)

        # Zero out diagonal (correct predictions)
        np.fill_diagonal(cm, 0)

        # Find top misclassifications
        misclass_pairs = []
        for true_idx in range(len(self.class_names)):
            for pred_idx in range(len(self.class_names)):
                if cm[true_idx, pred_idx] > 0:
                    misclass_pairs.append({
                        'true': self.class_names[true_idx],
                        'predicted': self.class_names[pred_idx],
                        'count': cm[true_idx, pred_idx]
                    })

        # Sort and get top N
        misclass_pairs = sorted(
            misclass_pairs,
            key=lambda x: x['count'],
            reverse=True
        )[:top_n]

        # Plot
        if misclass_pairs:
            labels = [
                f"{p['true'][:15]}\n→ {p['predicted'][:15]}"
                for p in misclass_pairs
            ]
            counts = [p['count'] for p in misclass_pairs]

            plt.figure(figsize=(12, 6))
            bars = plt.barh(range(len(labels)), counts, alpha=0.8)
            plt.yticks(range(len(labels)), labels, fontsize=9)
            plt.xlabel('Number of Misclassifications')
            plt.title(f'Top {top_n} Misclassification Pairs')
            plt.gca().invert_yaxis()

            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, counts)):
                plt.text(count, i, f' {count}', va='center')

            plt.tight_layout()
            plt.savefig(
                self.run_dir / 'misclassification_analysis.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.show()
            print("Saved: misclassification_analysis.png")

    def generate_classification_report(self):
        """Generate and save detailed classification report."""
        report = classification_report(
            self.all_labels,
            self.all_predictions,
            target_names=self.class_names,
            digits=3,
            zero_division=0
        )

        # Save text report
        with open(self.run_dir / 'classification_report.txt', 'w') as f:
            f.write(report)

        print("\nClassification Report:")
        print(report)
        print(f"\nSaved: classification_report.txt")

    def _save_results(self, metrics: Dict, use_confidence_threshold: bool):
        """Save all results to disk."""
        # Save metrics as JSON
        with open(self.run_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save predictions
        results_df = pd.DataFrame({
            'true_label': [self.class_names[i] for i in self.all_labels],
            'predicted_label': [
                'not_sure' if uncertain else self.class_names[pred]
                for pred, uncertain in zip(self.all_predictions, self.uncertain_mask)
            ],
            'confidence': self.all_confidences,
            'is_uncertain': self.uncertain_mask,
            'correct': [
                False if uncertain else (label == pred)
                for label, pred, uncertain in zip(
                    self.all_labels,
                    self.all_predictions,
                    self.uncertain_mask
                )
            ]
        })

        # Add probability columns
        for i, class_name in enumerate(self.class_names):
            results_df[f'prob_{class_name}'] = self.all_probabilities[:, i]

        results_df.to_csv(self.run_dir / 'predictions.csv', index=False)

        print(f"\n✓ Results saved to: {self.run_dir}")

    def print_summary(self, metrics: Dict):
        """Print a formatted summary of results."""
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Top-3 Accuracy:   {metrics['top_3_accuracy']:.4f}")
        print(f"Top-5 Accuracy:   {metrics['top_5_accuracy']:.4f}")
        print(f"\nMacro Average:")
        print(f"  Precision: {metrics['macro_precision']:.4f}")
        print(f"  Recall:    {metrics['macro_recall']:.4f}")
        print(f"  F1 Score:  {metrics['macro_f1']:.4f}")
        print(f"\nWeighted Average:")
        print(f"  Precision: {metrics['weighted_precision']:.4f}")
        print(f"  Recall:    {metrics['weighted_recall']:.4f}")
        print(f"  F1 Score:  {metrics['weighted_f1']:.4f}")
        print("="*60 + "\n")

    def run_full_evaluation(self, test_loader: DataLoader):
        """Run complete evaluation pipeline with all visualizations."""
        print("Starting full evaluation pipeline...\n")

        # 1. Run evaluation
        metrics = self.evaluate(test_loader)

        # 2. Print summary
        self.print_summary(metrics)

        # 3. Generate classification report
        self.generate_classification_report()

        # 4. Generate all plots
        print("\nGenerating visualizations...")
        self.plot_confusion_matrix(normalize=False)
        self.plot_confusion_matrix(normalize=True)
        self.plot_per_class_metrics()
        self.plot_roc_curves()
        self.plot_top_k_accuracy()
        self.plot_misclassification_analysis()

        print("\n✓ Full evaluation complete!")
        return metrics

