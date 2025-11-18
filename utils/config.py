from data.class_mapper import ClassMapping
from typing import Dict
import logging

__all__ = [
    "PreProcessConfig",
]

class PreProcessConfig:
    """Configuration centralisée du préprocessing"""

    def __init__(
        self,
        config_dict,
        image_size: tuple[int, int] = (224, 224),
        normalize_mean: tuple = (0.485, 0.456, 0.406),
        normalize_std: tuple = (0.229, 0.224, 0.225),
        num_classes: int = 10,
        class_mapping: Dict[str, int] = None
    ):
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.num_classes = num_classes

        for key, value in config_dict.items():
            self._set_nested_attr(key, value)

        self.logger = logging.getLoggerClass()

        self.class_mapping = ClassMapping(class_mapping or {}, self.logger) # The 'or' means if equal to None then {}

    def _set_nested_attr(self, key, value):
        keys = key.split('.')
        for k in keys[:-1]:
            if not hasattr(self, k):
                setattr(self, k, PreProcessConfig({}))
            current = getattr(self, k)
        setattr(self, keys[-1], value)
