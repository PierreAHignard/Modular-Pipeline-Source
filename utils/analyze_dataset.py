"""
Module de test et analyse des datasets d'images.
Fourni des métriques complètes pour déboguer les problèmes de données.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

__all__ = [
    "DatasetAnalyzer",
    "run_full_analysis"
]


class DatasetAnalyzer:
    """Analyseur complet pour datasets d'images"""

    def __init__(self, dataloader: DataLoader, device: str = "cpu"):
        self.dataloader = dataloader
        self.device = device
        self.results = {}

    def analyze_class_distribution(self) -> Dict:
        """Analyse la distribution des classes"""
        print("\n" + "="*60)
        print("📊 ANALYSE DE LA DISTRIBUTION DES CLASSES")
        print("="*60)

        labels_list = []

        for batch in tqdm(self.dataloader, desc="Collecte des labels"):
            _, labels = batch
            if isinstance(labels, torch.Tensor):
                labels_list.extend(labels.cpu().numpy().tolist())
            else:
                labels_list.extend(labels)

        label_counts = Counter(labels_list)
        total_samples = len(labels_list)

        # Statistiques
        results = {
            "total_samples": total_samples,
            "num_classes": len(label_counts),
            "class_distribution": dict(label_counts),
            "class_percentages": {
                cls: (count / total_samples * 100)
                for cls, count in label_counts.items()
            },
            "min_samples_per_class": min(label_counts.values()),
            "max_samples_per_class": max(label_counts.values()),
            "mean_samples_per_class": np.mean(list(label_counts.values())),
            "std_samples_per_class": np.std(list(label_counts.values())),
        }

        # Affichage
        print(f"\n✓ Total d'échantillons: {total_samples}")
        print(f"✓ Nombre de classes: {len(label_counts)}")
        print(f"✓ Min/Max par classe: {results['min_samples_per_class']}/{results['max_samples_per_class']}")
        print(f"✓ Moyenne par classe: {results['mean_samples_per_class']:.2f} ± {results['std_samples_per_class']:.2f}")

        # Vérifier déséquilibre
        ratio = results['max_samples_per_class'] / results['min_samples_per_class']
        if ratio > 10:
            print(f"⚠️  DÉSÉQUILIBRE CRITIQUE: Ratio max/min = {ratio:.2f}x")
        elif ratio > 3:
            print(f"⚠️  Déséquilibre modéré: Ratio max/min = {ratio:.2f}x")
        else:
            print(f"✓ Distribution équilibrée")

        print("\nDistribution par classe:")
        for cls in sorted(label_counts.keys()):
            count = label_counts[cls]
            pct = results['class_percentages'][cls]
            bar_length = int(count / max(label_counts.values()) * 30)
            print(f"  {str(cls):20s} | {count:4d} ({pct:5.1f}%) {'█' * bar_length}")

        self.results["class_distribution"] = results
        return results

    def analyze_image_properties(self) -> Dict:
        """Analyse les propriétés des images"""
        print("\n" + "="*60)
        print("🖼️  ANALYSE DES PROPRIÉTÉS DES IMAGES")
        print("="*60)

        shapes = []
        dtypes = []
        value_ranges = []
        corrupted = 0

        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Analyse images")):
            images, _ = batch

            for img in images:
                try:
                    shapes.append(tuple(img.shape))
                    dtypes.append(img.dtype)
                    value_ranges.append((img.min().item(), img.max().item()))
                except Exception as e:
                    corrupted += 1

        if corrupted > 0:
            print(f"⚠️  {corrupted} images corrompues détectées!")

        shape_counts = Counter(shapes)
        dtype_counts = Counter(dtypes)

        results = {
            "total_analyzed": len(shapes),
            "corrupted": corrupted,
            "unique_shapes": len(shape_counts),
            "shape_distribution": dict(shape_counts),
            "dtypes": dict(dtype_counts),
            "min_pixel_value": min(v[0] for v in value_ranges),
            "max_pixel_value": max(v[1] for v in value_ranges),
        }

        print(f"\n✓ Images analysées: {len(shapes)}")
        print(f"✓ Images corrompues: {corrupted}")
        print(f"✓ Formes uniques: {len(shape_counts)}")

        if len(shape_counts) > 1:
            print("⚠️  ATTENTION: Formes d'images inconsistantes!")
            for shape, count in sorted(shape_counts.items()):
                print(f"  {shape}: {count} images")
        else:
            print(f"✓ Toutes les images ont la même forme: {list(shape_counts.keys())[0]}")

        print(f"\n✓ Types de données: {dict(dtype_counts)}")
        print(f"✓ Plage de valeurs pixel: [{results['min_pixel_value']:.3f}, {results['max_pixel_value']:.3f}]")

        self.results["image_properties"] = results
        return results

    def analyze_batch_statistics(self) -> Dict:
        """Analyse les statistiques des batches"""
        print("\n" + "="*60)
        print("📈 ANALYSE DES STATISTIQUES DES BATCHES")
        print("="*60)

        batch_sizes = []
        means = []
        stds = []
        mins = []
        maxs = []

        for batch in tqdm(self.dataloader, desc="Statistiques batches"):
            images, _ = batch
            batch_sizes.append(len(images))

            # Normaliser en float si nécessaire
            img_float = images.float() if images.dtype != torch.float32 else images

            means.append(img_float.mean().item())
            stds.append(img_float.std().item())
            mins.append(img_float.min().item())
            maxs.append(img_float.max().item())

        results = {
            "num_batches": len(batch_sizes),
            "batch_size_min": min(batch_sizes),
            "batch_size_max": max(batch_sizes),
            "batch_size_mean": np.mean(batch_sizes),
            "pixel_mean": np.mean(means),
            "pixel_std": np.mean(stds),
            "pixel_min": np.min(mins),
            "pixel_max": np.max(maxs),
        }

        print(f"\n✓ Nombre de batches: {results['num_batches']}")
        print(f"✓ Taille des batches: {results['batch_size_min']}-{results['batch_size_max']} "
              f"(moy: {results['batch_size_mean']:.1f})")
        print(f"✓ Moyenne pixel globale: {results['pixel_mean']:.3f} ± {results['pixel_std']:.3f}")
        print(f"✓ Min/Max pixels: {results['pixel_min']:.3f} / {results['pixel_max']:.3f}")

        self.results["batch_statistics"] = results
        return results

    def check_label_types(self) -> Dict:
        """Vérifie les types et formats des labels"""
        print("\n" + "="*60)
        print("🏷️  VÉRIFICATION DES LABELS")
        print("="*60)

        label_types = {}
        first_batch_labels = None
        errors = []

        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Vérification labels")):
            _, labels = batch

            if batch_idx == 0:
                first_batch_labels = labels
                print(f"\n✓ Type des labels: {type(labels)}")
                print(f"✓ Shape: {labels.shape if hasattr(labels, 'shape') else 'N/A'}")
                if isinstance(labels, torch.Tensor):
                    print(f"✓ Dtype: {labels.dtype}")

                # Afficher quelques exemples
                print(f"\nExemples de labels (premiers 5):")
                for i in range(min(5, len(labels))):
                    print(f"  {i}: {labels[i]}")

            # Compter les types
            for label in labels:
                label_type = type(label).__name__
                if label_type not in label_types:
                    label_types[label_type] = 0
                label_types[label_type] += 1

            if batch_idx >= 2:  # Checker seulement les premiers batches
                break

        results = {
            "label_types": label_types,
            "first_batch": first_batch_labels,
        }

        print(f"\n✓ Types de labels trouvés: {label_types}")

        self.results["label_types"] = results
        return results

    def check_image_values(self) -> Dict:
        """Vérifie si les valeurs des images sont correctes"""
        print("\n" + "="*60)
        print("🔍 VÉRIFICATION DES VALEURS D'IMAGES")
        print("="*60)

        issues = {
            "all_zeros": 0,
            "all_ones": 0,
            "all_same_value": 0,
            "nan_values": 0,
            "inf_values": 0,
            "out_of_range": 0,
        }

        for batch in tqdm(self.dataloader, desc="Vérification valeurs"):
            images, _ = batch

            for img in images:
                img_float = img.float() if img.dtype != torch.float32 else img

                if torch.all(img_float == 0):
                    issues["all_zeros"] += 1
                elif torch.all(img_float == 1):
                    issues["all_ones"] += 1
                elif len(torch.unique(img_float)) == 1:
                    issues["all_same_value"] += 1

                if torch.isnan(img_float).any():
                    issues["nan_values"] += 1

                if torch.isinf(img_float).any():
                    issues["inf_values"] += 1

                if (img_float < 0).any() or (img_float > 1).any():
                    if torch.max(img_float) > 255 or torch.min(img_float) < 0:
                        issues["out_of_range"] += 1

        print("\n⚠️  Problèmes détectés:")
        for issue, count in issues.items():
            if count > 0:
                print(f"  ⚠️  {issue}: {count}")
            else:
                print(f"  ✓ {issue}: 0")

        self.results["value_issues"] = issues
        return issues

    def visualize_sample_batch(self, num_samples: int = 16, save_path: Optional[Path] = None):
        """Visualise un batch d'échantillons"""
        print("\n" + "="*60)
        print("📸 VISUALISATION D'UN BATCH")
        print("="*60)

        batch = next(iter(self.dataloader))
        images, labels = batch

        # Prendre les premiers num_samples
        num_show = min(num_samples, len(images))

        # Calculer la grille
        grid_size = int(np.ceil(np.sqrt(num_show)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()

        for idx in range(num_show):
            ax = axes[idx]

            img = images[idx]
            label = labels[idx]

            # Convertir en numpy pour affichage
            if isinstance(img, torch.Tensor):
                # Supposer format CHW
                if img.shape[0] == 3:  # RGB
                    img_np = img.permute(1, 2, 0).numpy()
                    if img_np.max() <= 1:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)
                else:  # Grayscale ou autre
                    img_np = img[0].numpy()
                    if img_np.max() <= 1:
                        img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = np.array(img)

            ax.imshow(img_np)
            ax.set_title(f"Label: {label}")
            ax.axis("off")

        # Cacher les axes inutilisés
        for idx in range(num_show, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            print(f"\n✓ Visualisation sauvegardée: {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Génère un rapport complet"""
        print("\n" + "="*60)
        print("📋 RAPPORT COMPLET")
        print("="*60)

        report = []
        report.append("="*60)
        report.append("RAPPORT D'ANALYSE DU DATASET")
        report.append("="*60)

        for section_name, section_data in self.results.items():
            report.append(f"\n{section_name.upper()}:")
            report.append("-" * 40)

            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if isinstance(value, dict):
                        report.append(f"  {key}:")
                        for k, v in value.items():
                            report.append(f"    {k}: {v}")
                    else:
                        report.append(f"  {key}: {value}")

        report_text = "\n".join(report)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report_text)
            print(f"\n✓ Rapport sauvegardé: {output_path}")

        return report_text


def run_full_analysis(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    output_dir: Optional[Path] = None,
    visualize: bool = True
):
    """Lance l'analyse complète du dataset"""

    output_dir = Path(output_dir) if output_dir else Path("./dataset_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "🚀"*30)
    print("DÉMARRAGE DE L'ANALYSE COMPLÈTE DU DATASET")
    print("🚀"*30)

    # Analyse Train
    print("\n\n📍 ANALYSE DU DATASET D'ENTRAÎNEMENT")
    train_analyzer = DatasetAnalyzer(train_loader)
    train_analyzer.analyze_class_distribution()
    train_analyzer.analyze_image_properties()
    train_analyzer.analyze_batch_statistics()
    train_analyzer.check_label_types()
    train_analyzer.check_image_values()

    if visualize:
        train_analyzer.visualize_sample_batch(
            num_samples=16,
            save_path=output_dir / "train_samples.png"
        )

    train_report = train_analyzer.generate_report(output_dir / "train_analysis.txt")

    # Analyse Validation (si fourni)
    if val_loader:
        print("\n\n📍 ANALYSE DU DATASET DE VALIDATION")
        val_analyzer = DatasetAnalyzer(val_loader)
        val_analyzer.analyze_class_distribution()
        val_analyzer.analyze_image_properties()
        val_analyzer.analyze_batch_statistics()
        val_analyzer.check_label_types()
        val_analyzer.check_image_values()

        if visualize:
            val_analyzer.visualize_sample_batch(
                num_samples=16,
                save_path=output_dir / "val_samples.png"
            )

        val_report = val_analyzer.generate_report(output_dir / "val_analysis.txt")

    print("\n" + "✓"*30)
    print("✓ ANALYSE TERMINÉE!")
    print("✓"*30)
    print(f"\nRésultats sauvegardés dans: {output_dir}")
