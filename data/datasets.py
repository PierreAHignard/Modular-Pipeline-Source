from abc import abstractmethod
from typing import Set, Optional, Callable, Any, Tuple, List, Dict
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

from utils.config import Config


__all__ = [
    "LocalImageDataset",
    "HuggingFaceImageDataset"
]

class CustomDataset(Dataset):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def labels(self):
        pass

class LocalImageDataset(CustomDataset):
    """
    Dataset pour images locales avec structure dossier = label.
    Version optimisée avec pré-chargement.
    """

    def __init__(
        self,
        data_dir: Path,
        config: Config,
        transforms: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
        preload: bool = True  # Nouveau paramètre
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.extensions = extensions
        self.config = config
        self.preload = preload

        # Chargement des paths
        self.samples = []
        self._load_samples()

        # Pré-chargement optionnel
        if self.preload:
            print("🔄 Pré-chargement des images en mémoire...")
            self._preload_images()

    def _load_samples(self):
        """Charge tous les chemins d'images et leurs labels"""
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        if not class_dirs:
            raise ValueError(f"Aucun sous-dossier trouvé dans {self.data_dir}")

        for class_dir in tqdm(class_dirs, desc="Scan des dossiers"):
            label = class_dir.name

            for ext in self.extensions:
                for img_path in class_dir.glob(f'*{ext}'):
                    self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(
                f"Aucune image trouvée dans {self.data_dir} "
                f"avec les extensions {self.extensions}"
            )

        print(f"✅ {len(self.samples)} images trouvées")

    def _preload_images(self):
        """Pré-charge toutes les images en mémoire"""
        self.preloaded_images = []

        for img_path, _ in tqdm(self.samples, desc="Chargement", unit="img"):
            image = Image.open(img_path).convert('RGB')
            self.preloaded_images.append(image.copy())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img_path, label = self.samples[idx]

        # Utiliser l'image pré-chargée ou charger à la volée
        if self.preload:
            image = self.preloaded_images[idx].copy()
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        label_idx = self.config.class_mapping[label]
        return image, label_idx

    @property
    def labels(self) -> Set[str]:
        return set(item[1] for item in self.samples)


class HuggingFaceImageDataset(CustomDataset):
    """
    Version optimisée avec pré-chargement complet en mémoire.
    Idéal pour environnements avec RAM suffisante (Kaggle, Colab).
    """

    def __init__(
        self,
        hf_dataset,
        config: Config,
        transforms: Optional[Callable] = None,
        image_column: str = 'image',
        label_column: str = 'label',
        bbox_column: Optional[str] = None,
        multi_label: bool = False
    ):
        super().__init__()

        self._cached_labels = None
        self.config = config
        self.transforms = transforms
        self.image_column = image_column
        self.label_column = label_column
        self.bbox_column = bbox_column
        self.multi_label = multi_label

        # Pré-chargement complet en mémoire
        print(f"🔄 Chargement du dataset en mémoire...")
        self._preload_dataset(hf_dataset)

        # Construction de l'index multi-label
        self._build_index()

        print(f"✅ Dataset prêt: {len(self.images)} images → {len(self)} échantillons")

    def _preload_dataset(self, hf_dataset):
        """Charge tout le dataset en mémoire de manière optimisée"""
        dataset_size = len(hf_dataset)

        # Pré-allocation des listes pour performance
        self.images: List[Image.Image] = []
        self.labels_list: List[Any] = []
        self.bboxes_list: Optional[List[Any]] = [] if self.bbox_column else None

        # Détection du mode multi-label sur le premier échantillon
        if dataset_size > 0:
            first_item = hf_dataset[0]
            first_labels = first_item[self.label_column]
            self.multi_label = isinstance(first_labels, (list, tuple))

        # Chargement avec barre de progression
        for item in tqdm(hf_dataset, desc="Chargement", total=dataset_size, unit="img"):
            # Image
            image = item[self.image_column]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image)).convert('RGB')
            self.images.append(image)

            # Labels
            labels = item[self.label_column]
            # Normaliser en liste pour traitement uniforme
            if not isinstance(labels, (list, tuple)):
                labels = [labels]
            self.labels_list.append(labels)

            # Bounding boxes (optionnel)
            if self.bbox_column and self.bbox_column in item:
                self.bboxes_list.append(item[self.bbox_column])
            elif self.bboxes_list is not None:
                self.bboxes_list.append(None)

    def _build_index(self):
        """
        Construit l'index de mapping (idx → (img_idx, label_idx))
        En multi-label: chaque couple (image, label) devient un échantillon
        """
        self.index_mapping: List[Tuple[int, int]] = []

        if self.multi_label:
            for img_idx, labels in enumerate(self.labels_list):
                for label_idx in range(len(labels)):
                    self.index_mapping.append((img_idx, label_idx))

            print(f"  → Mode multi-label détecté")
        else:
            # Mode single-label: mapping direct
            self.index_mapping = [(i, 0) for i in range(len(self.images))]

    def __len__(self) -> int:
        return len(self.index_mapping)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Accès ultra-rapide depuis la mémoire"""
        img_idx, label_idx = self.index_mapping[idx]

        # Récupération directe depuis les listes pré-chargées
        image = self.images[img_idx]
        labels = self.labels_list[img_idx]
        label = labels[label_idx]

        # Récupération de la bbox si disponible
        bbox = None
        if self.bboxes_list is not None and self.bboxes_list[img_idx] is not None:
            bboxes = self.bboxes_list[img_idx]
            if isinstance(bboxes, (list, tuple)) and len(bboxes) > label_idx:
                bbox = bboxes[label_idx]

        # Copie de l'image pour éviter les modifications en place
        # Important si vous utilisez des augmentations aléatoires
        image = image.copy()

        # Application des transformations
        if self.transforms:
            if bbox is not None:
                image = self.transforms(image, bbox=bbox)
            else:
                image = self.transforms(image)

        # Mapping du label vers l'index de classe
        label_idx_mapped = self.config.class_mapping[label]

        return image, label_idx_mapped

    @property
    def labels(self) -> Set[str]:
        """Retourne l'ensemble unique de tous les labels"""
        if self._cached_labels is None:
            all_labels = set()
            for labels in self.labels_list:
                all_labels.update(labels)
            self._cached_labels = all_labels
        return self._cached_labels

    def get_label_distribution(self) -> Dict[str, int]:
        """Retourne la distribution des labels (utile pour debug)"""
        from collections import Counter
        all_labels = []
        for labels in self.labels_list:
            all_labels.extend(labels)
        return dict(Counter(all_labels))

    def __repr__(self) -> str:
        return (f"HuggingFaceImageDataset("
                f"images={len(self.images)}, "
                f"samples={len(self)}, "
                f"multi_label={self.multi_label})")

