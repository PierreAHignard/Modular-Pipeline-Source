from abc import abstractmethod
from typing import Set, Optional, Callable, Any, Tuple, List, Union
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
    Dataset pour images locales avec structure dossier = label

    Args:
        - data_dir : Chemin vers le dossier racine contenant les sous-dossiers de classes
        - transforms : Transformations à appliquer
        - extensions : Extensions de fichiers images acceptées
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

        # Collecter tous les fichiers images et leurs labels
        self.samples = []

        self._load_samples()

    def _load_samples(self):
        """Charge tous les chemins d'images et leurs labels"""
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        n_classes = 0

        if not class_dirs:
            raise ValueError(f"Aucun sous-dossier trouvé dans {self.data_dir}")

        # Collecter tous les fichiers
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

        # Charger l'image
        image = Image.open(img_path).convert('RGB')

        if self.transforms:
            image, label = self.transforms(image, label)

        label = self.config.class_mapping[label]

        return image, label

    @property
    def labels(self) -> Set[str]:
        """Donne le Set des labels uniques présents dans le dataset"""
        return set(item[1] for item in self.samples)

class HuggingFaceImageDataset(CustomDataset):
    """Wrapper pour dataset HuggingFace avec colonnes 'image' et 'label'"""

    def __init__(
        self,
        hf_dataset,
        config: Config,
        transforms: Optional[Callable] = None,
        image_column: str = 'image',
        label_column: str = 'label'
    ):
        self.dataset = hf_dataset
        self.config = config
        self.transforms = transforms
        self.image_column = image_column
        self.label_column = label_column

        super().__init__()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        item = self.dataset[idx]

        image = item[self.image_column]
        label = item[self.label_column][0] #TODO : [0] is a temporary fix for multi-label

        # Convertir en PIL Image si nécessaire
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image)).convert('RGB')

        if self.transforms:
            image, label = self.transforms(image, label)

        label = self.config.class_mapping[label]

        return image, label

    @property
    def labels(self) -> Set[str]:
        """Donne le Set des labels uniques présents dans le dataset"""
        return set(item[self.label_column][0] for item in self.dataset) #TODO : [0] is a temporary fix for multi-label

