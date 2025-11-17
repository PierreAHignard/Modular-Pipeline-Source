from typing import Dict, Any, Tuple
from src.preprocessing.transforms import PreProcessor, DataAugmentation, ModelSpecificPreprocessor
from src.data.config import PreProcessConfig

# ============ COMPOSITION DES TRANSFORMATIONS ============

class TransformPipeline:
    """Pipeline complète : combine les 3 étapes dans le bon ordre"""

    def __init__(
        self,
        config: PreProcessConfig,
        model_name: str = 'resnet50',
        is_train: bool = True,
        class_mapping: Dict[str, int] = None
    ):
        self.config = config
        self.is_train = is_train
        self.class_mapping = class_mapping or {}

        # Instancier dans l'ordre
        self.pre_processor = PreProcessor(config)
        self.augmentor = DataAugmentation(config, is_train=is_train)
        self.model_processor = ModelSpecificPreprocessor(config, model_name)

    def set_class_mapping(self, mapping: Dict[str, int]):
        """Définir le mapping str -> int pour les labels"""
        self.pre_processor._label_to_idx = lambda label: mapping.get(label, 0)

    def __call__(self, image: Any, label: str) -> Tuple[str, Any]:
        # Étape 1
        image, label = self.pre_processor(image, label)

        # Étape 2
        image, label = self.augmentor(image, label)

        # Étape 3
        image, label = self.model_processor(image, label)

        return image, label
