import torch
import torchvision.transforms as T
from typing import Dict, Any, Tuple
from utils.config import Config
from PIL import Image

# ============ ÉTAPE 1: PRE_PREPROCESS ============

class PreProcessor:
    """Étape 1: Standardisation des données"""

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, image: Any, label: str) -> Any:
        # Traiter l'image
        # Convertir en PIL si nécessaire
        if isinstance(image, str):
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

        # AVANT le mapping
        #print("------------Testing for issues with mapping-----------")
        #print(self.config.class_mapping)
        #print(f"Wants to map label : '{label}'")
        #print("Class mapping before:", self.config.class_mapping.mapping)
        #print("Class size before:", self.config.class_mapping.size)

        # Le mapping se crée ici automatiquement
        label = self.config.class_mapping[label]

        # APRÈS le mapping
        #print("Class mapping after:", self.config.class_mapping.mapping)
        #print("Class size after:", self.config.class_mapping.size)
        #print(f"Label '{label}' mapped to ID: {label}")

        return image, label


# ============ ÉTAPE 2: DATA AUGMENTATION ============

class DataAugmentation:
    """Étape 2: Augmentation des données (seulement train)"""

    def __init__(self, config: Config, is_train: bool = True):
        self.config = config
        self.is_train = is_train

        if is_train:
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
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

    def __init__(self, config: Config, model_name: str = 'resnet50'):
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
