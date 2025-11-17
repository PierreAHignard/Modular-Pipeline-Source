import torch
import torchvision.transforms as T
from typing import Dict, Any, Tuple
from src.data.config import PreProcessConfig



# ============ ÉTAPE 1: PRE_PREPROCESS ============

class PreProcessor:
    """Étape 1: Standardisation des données"""

    def __init__(self, config: PreProcessConfig):
        self.config = config

    def __call__(self, image: Any, label: str) -> Any:
        # Traiter l'image
        # Convertir en PIL si nécessaire
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image)

        # Convertir en RGB (supprimer RGBA si présent)
        if image.mode in ['RGBA', 'LA']:
            rgb = Image.new('RGB', image.size, (255, 255, 255))
            rgb.paste(image, mask=image.split()[-1] if 'A' in image.mode else None)
            image = rgb
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        # Redimensionner
        image = image.resize(self.config.image_size)

        # Convertir en tensor float32
        image = T.ToTensor()(image)

        # Traiter les labels : str -> idx
        if isinstance(label, str):
            # Supposant une liste de classes en dehors
            label = self._label_to_idx(label)

        return image, label

    def _label_to_idx(self, label_str: str) -> int:
        """À customiser selon votre liste de classes"""
        raise NotImplementedError("Implémenter le mapping label_str -> idx")


# ============ ÉTAPE 2: DATA AUGMENTATION ============

class DataAugmentation:
    """Étape 2: Augmentation des données (seulement train)"""

    def __init__(self, config: PreProcessConfig, is_train: bool = True):
        self.config = config
        self.is_train = is_train

        if is_train:
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ])
        else:
            self.transforms = T.Compose([])  # Pas d'augmentation en eval

    def __call__(self, image: Any, label: str) -> Tuple[str, Any]:
        if self.is_train:
            # Convertir tensor -> PIL pour torchvision.transforms
            if isinstance(image, torch.Tensor):
                image = T.ToPILImage()(image)

            # Appliquer augmentations
            image = self.transforms(image)

            # Reconvertir en tensor
            image = T.ToTensor()(image)

        return image, label


# ============ ÉTAPE 3: MODEL-SPECIFIC PREPROCESS ============

class ModelSpecificPreprocessor:
    """Étape 3: Normalisation spécifique au modèle"""

    def __init__(self, config: PreProcessConfig, model_name: str = 'resnet50'):
        self.config = config
        self.model_name = model_name

        # Normalisation ImageNet standard pour plupart des modèles
        self.normalize = T.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std
        )

    def __call__(self, image: Any, label: str) -> Tuple[str, Any]:
        # S'assurer que c'est un tensor
        if not isinstance(image, torch.Tensor):
            image = T.ToTensor()(image)

        # S'assurer que c'est 3 canaux (C, H, W)
        if image.dim() == 2:  # Grayscale
            image = image.unsqueeze(0).repeat(3, 1, 1)

        # Normaliser
        image = self.normalize(image)

        return image, label
