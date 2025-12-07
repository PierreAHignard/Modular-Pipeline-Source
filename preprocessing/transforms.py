import torch
import torchvision.transforms as T
from typing import Any, Tuple, Optional
from utils.config import Config
from PIL import Image

# ============ CROP PAR BOUNDING BOX ============

class CropByBoundingBox:
    """Crop l'image selon une bounding box avec padding et aspect ratio"""

    def __init__(self, padding: float = 0.0, target_size: Optional[Tuple[int, int]] = None):
        """
        Args:
            padding: Pourcentage de padding à ajouter autour de la bbox (0.1 = 10%)
            target_size: (width, height) cible pour maintenir l'aspect ratio, ex: (224, 224)
        """
        self.padding = padding
        self.target_size = target_size
        self.target_ratio = target_size[0] / target_size[1] if target_size else 1

    def __call__(self, image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
        """
        Args:
            image: Image PIL
            bbox: (x, y, width, height) en pixels ou normalisées [0,1]
        """
        if bbox is None:
            return image

        img_width, img_height = image.size
        x, y, w, h = bbox

        # Convertir coordonnées normalisées si besoin
        if w <= 2 and h <= 2:
            x = int(x * img_width)
            y = int(y * img_height)
            w = int(w * img_width)
            h = int(h * img_height)

        # Calculer le centre de la bbox
        center_x = x + w / 2
        center_y = y + h / 2

        # Ajouter padding
        if self.padding > 0:
            w = w * (1 + 2 * self.padding)
            h = h * (1 + 2 * self.padding)

        # Ajuster l'aspect ratio si nécessaire
        if self.target_ratio:
            current_ratio = w / h
            if current_ratio > self.target_ratio:
                h = w / self.target_ratio
            else:
                w = h * self.target_ratio

        # Calculer les coordonnées souhaitées
        x1 = center_x - w / 2
        y1 = center_y - h / 2
        x2 = center_x + w / 2
        y2 = center_y + h / 2

        # Gérer les débordements en déplaçant le centre
        if x1 < 0:
            shift = -x1
            x1 += shift
            x2 += shift
        elif x2 > img_width:
            shift = x2 - img_width
            x1 -= shift
            x2 -= shift

        if y1 < 0:
            shift = -y1
            y1 += shift
            y2 += shift
        elif y2 > img_height:
            shift = y2 - img_height
            y1 -= shift
            y2 -= shift

        # Si après déplacement ça déborde encore, réduire la taille (cas extrême)
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img_width, int(x2))
        y2 = min(img_height, int(y2))

        # Crop
        return image.crop((x1, y1, x2, y2))


# ============ ÉTAPE 1: PRE_PREPROCESS ============

class PreProcessor:
    """Étape 1: Standardisation des données"""

    def __init__(self, config: Config, bbox_padding: float = 0.05):
        self.config = config
        self.crop_bbox = CropByBoundingBox(padding=bbox_padding, target_size=self.config.image_size)

    def __call__(self, image: Any, bbox: Any = None) -> Any:
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

        # Crop l'image s'il y a des coordonnées de bbox
        if bbox is not None:
            image = self.crop_bbox(image, bbox)

        # Redimensionner
        image = image.resize(self.config.image_size)

        # Convertir en tensor float32
        image = T.ToTensor()(image)

        return image


# ============ ÉTAPE 2: DATA AUGMENTATION ============

class DataAugmentation:
    """Étape 2: Augmentation des données (seulement train)"""

    def __init__(self, config: Config, is_train: bool = True):
        self.config = config
        self.is_train = is_train

        if is_train:
            transforms = self.config.transforms
        else:
            transforms = [] # Pas d'augmentation en eval

        self.transforms = T.Compose(transforms)

    def __call__(self, image: Any) -> Tuple[str, Any]:
        if self.is_train:
            # Convertir tensor -> PIL pour torchvision.transforms
            if isinstance(image, torch.Tensor):
                image = T.ToPILImage()(image)

            # Appliquer augmentations
            image = self.transforms(image)

            # Reconvertir en tensor
            image = T.ToTensor()(image)

        return image


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

    def __call__(self, image: Any) -> Tuple[str, Any]:
        # S'assurer que c'est un tensor
        if not isinstance(image, torch.Tensor):
            image = T.ToTensor()(image)

        # S'assurer que c'est 3 canaux (C, H, W)
        if image.dim() == 2:  # Grayscale
            image = image.unsqueeze(0).repeat(3, 1, 1)

        # Normaliser
        image = self.normalize(image)

        return image
