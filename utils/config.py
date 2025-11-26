from data.class_mapper import ClassMapping
from typing import Dict
import logging
import torch

__all__ = [
    "Config",
]

class Config:
    """Configuration centralisée du préprocessing"""

    def __init__(
        self,
        config_dict,
        logger: logging.Logger = None,
        image_size: tuple[int, int] = (224, 224),
        normalize_mean: tuple = (0.485, 0.456, 0.406),
        normalize_std: tuple = (0.229, 0.224, 0.225),
        class_mapping: Dict[str, int] = None,
        allow_new_class_outside_preload: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.device = device

        for key, value in config_dict.items():
            self._set_nested_attr(key, value)

        self.logger = logger or logging.getLogger(__name__)

        self.class_mapping = ClassMapping(class_mapping, self.logger, allow_new_class_outside_preload)

    def _set_nested_attr(self, key, value):
        keys = key.split('.')
        current = self

        # Créer/naviguer à travers tous les niveaux sauf le dernier
        for k in keys[:-1]:
            if not hasattr(current, k) or getattr(current, k) is None:
                setattr(current, k, Config({}))
            current = getattr(current, k)

        # Définir la valeur au dernier niveau
        setattr(current, keys[-1], value)