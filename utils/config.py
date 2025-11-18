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

        self.logger = logging.getLoggerClass()

        self.class_mapping = ClassMapping(class_mapping or {}, self.logger) # The 'or' means if equal to None then {}
