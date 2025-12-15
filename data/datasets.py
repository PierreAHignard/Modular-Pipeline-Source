from abc import abstractmethod
from typing import Set, Optional, Callable, Any, Tuple, List, Dict
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from preprocessing.transforms import PreProcessor
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

    @abstractmethod
    def get_label_distribution(self):
        pass


class LocalImageDataset(CustomDataset):
    """
    Dataset pour images locales avec structure dossier = label.
    Version optimisée avec pré-chargement.
    """

    def __init__(
            self,
            data_dir: Path,
            config: Config,
            transforms: Optional[Callable] = None,
            extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
            preload: bool = True,
            # NOUVEAU PARAMÈTRE
            discard_unmapped_classes: bool = False
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.extensions = extensions
        self.config = config
        self.preload = preload
        # NOUVEAU
        self.discard_unmapped_classes = discard_unmapped_classes
        self._discarded_classes_count = 0  # Compteur pour le résumé

        # Chargement des paths
        self.samples = []
        self._load_samples()

        # Pré-chargement optionnel
        if self.preload:
            print("🔄 Pré-chargement des images en mémoire...")
            self._preload_images()

        # Affichage du résumé
        if self.discard_unmapped_classes and self._discarded_classes_count > 0:
            print(
                f"⚠️ {self._discarded_classes_count} échantillons ignorés car leur classe n'était pas dans class_mapping.")

    def _load_samples(self):
        """Charge tous les chemins d'images et leurs labels"""
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        # Récupération des labels mappés pour vérification rapide
        mapped_labels = set(self.config.class_mapping.labels)

        if not class_dirs:
            raise ValueError(f"Aucun sous-dossier trouvé dans {self.data_dir}")

        for class_dir in tqdm(class_dirs, desc="Scan des dossiers"):
            label = class_dir.name

            # NOUVEAU: Vérifier si la classe doit être ignorée
            if self.discard_unmapped_classes and label not in mapped_labels:

                # Compter les images ignorées dans ce dossier
                count_in_dir = 0
                for ext in self.extensions:
                    count_in_dir += len(list(class_dir.glob(f'*{ext}')))

                self._discarded_classes_count += count_in_dir

                print(f"⚠️ Classe '{label}' ignorée ({count_in_dir} images). Non présente dans class_mapping.")
                continue  # Passer à la classe suivante

            for ext in self.extensions:
                for img_path in class_dir.glob(f'*{ext}'):
                    self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(
                f"Aucune image trouvée dans {self.data_dir} "
                f"avec les extensions {self.extensions}. "
                "Vérifiez les dossiers et le class_mapping."
            )

        print(f"✅ {len(self.samples)} images trouvées")

    def _preload_images(self):
        """Pré-charge toutes les images en mémoire"""
        self.preloaded_images = []

        for img_path, _ in tqdm(self.samples, desc="Chargement", unit="img"):
            image = Image.open(img_path).convert('RGB')
            self.preloaded_images.append(image.copy())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img_path, label = self.samples[idx]

        # Utiliser l'image pré-chargée ou charger à la volée
        if self.preload:
            image = self.preloaded_images[idx].copy()
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        # Utilisation de la méthode __getitem__ du ClassMapping pour la récupération sécurisée
        label_idx = self.config.class_mapping[label]
        return image, label_idx

    @property
    def labels(self) -> Set[str]:
        return set(item[1] for item in self.samples)

    def get_label_distribution(self):
        raise NotImplementedError


class HuggingFaceImageDataset(CustomDataset):
    """
    Version optimisée avec pré-chargement complet en mémoire.
    Idéal pour environnements avec RAM suffisante (Kaggle, Colab).
    """

    def __init__(
            self,
            hf_dataset,
            config: Config,
            transforms: Optional[Callable] = None,
            image_column: str = 'image',
            label_column: str = 'label',
            bbox_column: str = 'boxes',
            multi_label: bool = False,
            bbox_padding: float = 0.1,
            discard_unmapped_classes: bool = False
    ):
        super().__init__()

        self.config = config
        self.transforms = transforms
        self.image_column = image_column
        self.label_column = label_column
        self.bbox_column = bbox_column
        self.multi_label = multi_label
        self.discard_unmapped_classes = discard_unmapped_classes

        self.pre_processor = PreProcessor(config, bbox_padding=bbox_padding)

        # Pré-chargement complet en mémoire
        print(f"🔄 Chargement du dataset en mémoire...")
        self.discarded_count = 0  # Compteur d'échantillons ignorés
        self._preload_dataset(hf_dataset)

        # Affichage du résumé
        print(f"✅ Dataset prêt: {len(hf_dataset)} images → {len(self)} échantillons")
        if self.discarded_count > 0:
            print(
                f"⚠️ {self.discarded_count} échantillons (label/image) ignorés car leur classe n'était pas dans class_mapping.")

    def _preload_dataset(self, hf_dataset):
        """Charge tout le dataset en mémoire de manière optimisée"""
        dataset_size = len(hf_dataset)

        # Pré-allocation des listes pour performance
        self.images_tensor: List[Tensor] = []
        self.label_list: List[Any] = []

        # Récupération des labels mappés pour vérification rapide
        mapped_labels = set(self.config.class_mapping.labels)

        # Détection du mode multi-label sur le premier échantillon (logique conservée)
        if dataset_size > 0:
            first_item = hf_dataset[0]
            first_labels = first_item[self.label_column]
            self.multi_label = isinstance(first_labels, (list, tuple))

        # Chargement avec barre de progression
        for item in tqdm(hf_dataset, desc="Chargement", total=dataset_size, unit="img"):
            # Labels
            labels = item[self.label_column]
            # Normaliser en liste pour traitement uniforme
            if not isinstance(labels, (list, tuple)):
                labels = [labels]

            # Bounding boxes
            if self.bbox_column and self.bbox_column in item:
                bboxes = item[self.bbox_column]
            else:
                bboxes = [None] * len(labels)

            if len(labels) != len(bboxes):
                raise Exception(f"Number of labels ({len(labels)}) not equal to number of bboxes ({len(bboxes)}).")

            image = item[self.image_column]

            # Traitement des labels/bboxes
            for i in range(len(labels)):
                current_label = labels[i]
                current_bbox = bboxes[i]

                # Vérification de l'existence du label dans le mapping
                if self.discard_unmapped_classes and current_label not in mapped_labels:
                    self.discarded_count += 1
                    continue  # Ignorer cet échantillon (paire image/label)

                # Image
                cropped_image = self.pre_processor(image, bbox=current_bbox)
                self.images_tensor.append(cropped_image)

                # Label (doit exister dans le mapping car on a vérifié, on peut utiliser __getitem__)
                # IMPORTANT: Si vous voulez ajouter de nouveaux labels *après* le preload,
                # il faut utiliser .set(), sinon utilisez __getitem__ (qui est le comportement par défaut si non trouvé).
                # Ici, on utilise .set() pour s'assurer que si la classe a été conservée, elle est bien mappée.
                # Si discard_unmapped_classes est True, .set() ne sera appelé que pour les classes déjà mappées.
                # Si discard_unmapped_classes est False, .set() mappera les nouvelles classes.
                mapped_label = self.config.class_mapping.set(current_label)
                self.label_list.append(mapped_label)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Accès rapide depuis la mémoire"""

        # Récupération directe depuis les listes
        image = self.images_tensor[idx]
        label = self.label_list[idx]

        # Application des transformations
        if self.transforms:
            image = self.transforms(image)

        return image, label

    def __len__(self) -> int:
        return len(self.label_list)

    @property
    def labels(self) -> Set[str]:
        """Retourne l'ensemble unique de tous les labels mappés (indices)"""
        # Note: ceci retourne l'ensemble des INDICES, pas les noms de classes originales si elles n'étaient pas mappées initialement.
        return set(self.label_list)

    def get_label_distribution(self) -> Dict[str, int]:
        """Retourne la distribution des labels mappés"""
        from collections import Counter

        return dict(Counter(self.label_list))