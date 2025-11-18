from typing import Set, Optional, Callable, Any, Tuple, List, Union
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


__all__ = [
    "LocalImageDataset",
    "HuggingFaceImageDataset"
]

class LocalImageDataset(Dataset):
    """Dataset pour images locales avec structure dossier = label"""

    def __init__(
        self,
        data_dir: Path,
        transforms: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    ):
        """
        Args:
            data_dir: Chemin vers le dossier racine contenant les sous-dossiers de classes
            transforms: Transformations à appliquer
            extensions: Extensions de fichiers images acceptées
        """
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.extensions = extensions

        # Collecter tous les fichiers images et leurs labels
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        self._load_samples()

    def _load_samples(self):
        """Charge tous les chemins d'images et leurs labels"""
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        if not class_dirs:
            raise ValueError(f"Aucun sous-dossier trouvé dans {self.data_dir}")

        # Créer le mapping classe -> index
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name

        # Collecter tous les fichiers
        for class_dir in class_dirs:
            class_name = class_dir.name
            label = self.class_to_idx[class_name]

            for ext in self.extensions:
                for img_path in class_dir.glob(f'*{ext}'):
                    self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(
                f"Aucune image trouvée dans {self.data_dir} avec les extensions {self.extensions}"
            )

        print(f"Dataset chargé: {len(self.samples)} images, {len(self.class_to_idx)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        img_path, label = self.samples[idx]

        # Charger l'image
        image = Image.open(img_path).convert('RGB')

        if self.transforms:
            image, label = self.transforms(image, label)

        return image, label

    def get_labels(self) -> Set[str]:
        """Donne le Set des labels uniques présents dans le dataset"""
        return set(self.class_to_idx.keys())

class HuggingFaceImageDataset(Dataset):
    """Wrapper pour dataset HuggingFace avec colonnes 'image' et 'label'"""

    def __init__(
        self,
        hf_dataset,
        transforms: Optional[Callable] = None,
        image_column: str = 'image',
        label_column: str = 'label'
    ):
        self.dataset = hf_dataset
        self.transforms = transforms
        self.image_column = image_column
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        item = self.dataset[idx]

        image = item[self.image_column]
        label = item[self.label_column]

        # Convertir en PIL Image si nécessaire
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image)).convert('RGB')

        if self.transforms:
            image, label = self.transforms(image, label)

        return image, label

    def get_labels(self) -> Set[str]:
        """Donne le Set des labels uniques présents dans le dataset"""
        return set(item[self.label_column] for item in self.dataset)
