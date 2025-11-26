from typing import Dict, Any, Tuple
from preprocessing.transforms import PreProcessor, DataAugmentation, ModelSpecificPreprocessor
from utils.config import Config

# ============ COMPOSITION DES TRANSFORMATIONS ============

class TransformPipeline:
    """Pipeline complète : combine les 3 étapes dans le bon ordre"""

    def __init__(
        self,
        config: Config,
        model_name: str = 'resnet50',
        is_train: bool = True
    ):
        self.config = config
        self.is_train = is_train

        # Instancier dans l'ordre
        self.pre_processor = PreProcessor(config)
        self.augmentor = DataAugmentation(config, is_train=is_train)
        self.model_processor = ModelSpecificPreprocessor(config, model_name)

    def __call__(self, image: Any) -> Tuple[str, Any]:
        # Global Preprocessing
        image = self.pre_processor(image)

        # Data Augmentation
        image = self.augmentor(image)

        # Model-Specific Preprocessing
        image = self.model_processor(image)

        return image