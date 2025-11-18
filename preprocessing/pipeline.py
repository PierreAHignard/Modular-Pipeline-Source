from typing import Dict, Any, Tuple
from preprocessing.transforms import PreProcessor, DataAugmentation, ModelSpecificPreprocessor
from data.config import PreProcessConfig
from data.class_mapper import ClassMapping

# ============ COMPOSITION DES TRANSFORMATIONS ============

class TransformPipeline:
    """Pipeline complète : combine les 3 étapes dans le bon ordre
    Set class_mapping to None for automatic mapping"""

    def __init__(
        self,
        config: PreProcessConfig,
        model_name: str = 'resnet50',
        is_train: bool = True,
        class_mapping: Dict[str, int] = None
    ):
        self.config = config
        self.is_train = is_train
        self.class_mapping = ClassMapping(class_mapping)

        # Instancier dans l'ordre
        self.pre_processor = PreProcessor(config)
        self.augmentor = DataAugmentation(config, is_train=is_train)
        self.model_processor = ModelSpecificPreprocessor(config, model_name)

    def __call__(self, image: Any, label: str) -> Tuple[str, Any]:
        # Étape 1
        image, label = self.pre_processor(image, label)

        # Étape 2
        image, label = self.augmentor(image, label)

        # Étape 3
        image, label = self.model_processor(image, label)

        return image, label