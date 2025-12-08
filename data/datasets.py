from abc import abstractmethod
from typing import Set, Optional, Callable, Any, Tuple
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
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

    ⚠️ NE FONCTIONNE QU'AVEC DU SINGLE LABEL ⚠️
    """

    def __init__(
        self,
        data_dir: Path,
        config: Config,
        transforms: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    ):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.extensions = extensions
        self.config = config

        super().__init__()

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """Charge tous les chemins d'images et leurs labels"""
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        n_classes = 0

        if not class_dirs:
            raise ValueError(f"Aucun sous-dossier trouvé dans {self.data_dir}")

        for class_dir in class_dirs:
            label = class_dir.name
            n_classes += 1

            for ext in self.extensions:
                for img_path in class_dir.glob(f'*{ext}'):
                    self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(
                f"Aucune image trouvée dans {self.data_dir} avec les extensions {self.extensions}"
            )

        print(f"Dataset chargé: {len(self.samples)} images, {n_classes} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        label = self.config.class_mapping[label]
        return image, label

    @property
    def labels(self) -> Set[str]:
        return set(item[1] for item in self.samples)


class HuggingFaceImageDataset(CustomDataset):
    """
    Version optimisée pour environnements à RAM limitée (Kaggle, Colab)
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
        self.config = config
        self.transforms = transforms
        self.image_column = image_column
        self.label_column = label_column
        self.bbox_column = bbox_column
        self.multi_label = multi_label

        super().__init__()

        # Extraction UNIQUEMENT des métadonnées (légères)
        print("Extraction des métadonnées du dataset...")
        self.dataset = hf_dataset
        self._extract_metadata()
        self._build_index()

    def _extract_metadata(self):
        """Extrait seulement les infos nécessaires (labels, pas les images)"""
        # Accès optimisé aux colonnes HuggingFace
        # Charge UNIQUEMENT la colonne des labels, pas les images
        label_data = self.dataset[self.label_column]

        # Convertir en liste si c'est un IterableColumn
        if hasattr(label_data, '__iter__') and not isinstance(label_data, (list, str)):
            label_data = list(label_data)

        # Détection auto du mode multi-label
        if not self.multi_label and len(label_data) > 0:
            first_labels = label_data[0]
            self.multi_label = isinstance(first_labels, (list, tuple)) and len(first_labels) > 1

        # Stocker le nombre de labels par image (très léger en mémoire)
        self.n_labels_per_image = []
        for labels in label_data:
            if not isinstance(labels, (list, tuple)):
                labels = [labels]
            self.n_labels_per_image.append(len(labels))

        print(f"  → {len(self.n_labels_per_image)} images analysées")

    def _build_index(self):
        """Construit l'index à partir des métadonnées"""
        self.index_mapping = []

        if self.multi_label:
            for img_idx, n_labels in enumerate(self.n_labels_per_image):
                for label_idx in range(n_labels):
                    self.index_mapping.append((img_idx, label_idx))

            print(f"  ✓ Mode multi-label: {len(self.n_labels_per_image)} images → {len(self.index_mapping)} échantillons")
        else:
            self.index_mapping = [(i, 0) for i in range(len(self.n_labels_per_image))]

    def __len__(self) -> int:
        return len(self.index_mapping)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img_idx, label_idx = self.index_mapping[idx]
        item = self.dataset[img_idx]

        # Charger l'image
        image = item[self.image_column]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image)).convert('RGB')

        # Récupérer le label
        labels = item[self.label_column]
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        label = labels[label_idx]

        # Récupérer la bbox si disponible
        bbox = None
        if self.bbox_column and self.bbox_column in item:
            bboxes = item[self.bbox_column]
            if isinstance(bboxes, (list, tuple)) and len(bboxes) > label_idx:
                bbox = bboxes[label_idx]

        # Appliquer les transformations
        if self.transforms:
            image = self.transforms(image, bbox=bbox)

        # Mapper le label
        label = self.config.class_mapping[label]

        return image, label

    @property
    def labels(self) -> Set[str]:
        """Version optimisée qui accède seulement à la colonne labels"""
        if not hasattr(self, '_cached_labels'):
            all_labels = set()
            label_data = self.dataset[self.label_column]
            for labels in label_data:
                if isinstance(labels, (list, tuple)):
                    all_labels.update(labels)
                else:
                    all_labels.add(labels)
            self._cached_labels = all_labels
        return self._cached_labels
