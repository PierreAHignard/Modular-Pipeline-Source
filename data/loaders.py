"""
Module de chargement de datasets d'images.
Supports: Hugging Face, Local Files (structure dossier = label)
Permet de créer des splits train/val/test
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable, Any, Tuple, List, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
import numpy as np
from src.data.datasets import LocalImageDataset, HuggingFaceImageDataset

__all__ = [
    "BaseImageDatasetLoader",
    "LocalImageDatasetLoader",
    "HuggingFaceImageDatasetLoader",
]


class BaseImageDatasetLoader(ABC):
    """Classe de base abstraite pour tous les loaders de datasets d'images"""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
        transforms: Optional[Callable] = None,
        split_ratios: Optional[Tuple[float, ...]] = None,
        seed: int = 42
    ):
        """
        Args:
            batch_size: Taille des batches
            num_workers: Nombre de workers pour le DataLoader
            shuffle: Mélanger les données
            transforms: Transformations à appliquer aux images
            split_ratios: Ratios pour split (ex: (0.7, 0.15, 0.15) pour train/val/test)
                         ou (0.8, 0.2) pour train/val. Si None, pas de split.
            seed: Graine aléatoire pour la reproductibilité des splits
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.transforms = transforms
        self.split_ratios = split_ratios
        self.seed = seed

        if split_ratios is not None:
            if not (2 <= len(split_ratios) <= 3):
                raise ValueError("split_ratios doit contenir 2 ou 3 valeurs")
            if not np.isclose(sum(split_ratios), 1.0):
                raise ValueError(f"Les ratios doivent sommer à 1.0, actuellement: {sum(split_ratios)}")

    @abstractmethod
    def load_data(self) -> Dataset:
        """Charge et retourne le dataset"""
        pass

    def split_dataset(
        self,
        dataset: Dataset
    ) -> Union[Dataset, Tuple[Dataset, ...]]:
        """
        Split le dataset selon les ratios spécifiés

        Returns:
            Si split_ratios est None: retourne le dataset complet
            Si split_ratios: retourne tuple de (train, val) ou (train, val, test)
        """
        if self.split_ratios is None:
            return dataset

        total_size = len(dataset)
        sizes = [int(ratio * total_size) for ratio in self.split_ratios]

        # Ajuster le dernier pour éviter les erreurs d'arrondi
        sizes[-1] = total_size - sum(sizes[:-1])

        generator = torch.Generator().manual_seed(self.seed)
        splits = random_split(dataset, sizes, generator=generator)

        return tuple(splits)

    def get_dataloader(
        self,
        dataset: Optional[Dataset] = None,
        shuffle_override: Optional[bool] = None
    ) -> Union[DataLoader, Tuple[DataLoader, ...]]:
        """
        Crée un ou plusieurs DataLoader PyTorch

        Returns:
            Si pas de split: un DataLoader
            Si split: tuple de DataLoaders (train, val) ou (train, val, test)
        """
        if dataset is None:
            dataset = self.load_data()

        # Si le dataset est déjà un tuple (après split)
        if isinstance(dataset, tuple):
            loaders = []
            for i, ds in enumerate(dataset):
                # Par défaut: shuffle pour train, pas shuffle pour val/test
                shuffle = self.shuffle if i == 0 else False
                if shuffle_override is not None:
                    shuffle = shuffle_override

                loaders.append(DataLoader(
                    ds,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=shuffle,
                    pin_memory=torch.cuda.is_available()
                ))
            return tuple(loaders)

        # Sinon, split puis créer les loaders
        splits = self.split_dataset(dataset)
        if isinstance(splits, tuple):
            return self.get_dataloader(splits, shuffle_override)

        # Dataset simple sans split
        shuffle = self.shuffle if shuffle_override is None else shuffle_override
        return DataLoader(
            splits,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available()
        )


class LocalImageDatasetLoader(BaseImageDatasetLoader):
    """Loader pour datasets d'images locaux"""

    def __init__(
        self,
        data_dir: Path,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir)
        self.extensions = extensions

    def load_data(self) -> LocalImageDataset:
        """Charge le dataset local"""
        return LocalImageDataset(
            data_dir=self.data_dir,
            transforms=self.transforms,
            extensions=self.extensions
        )


class HuggingFaceImageDatasetLoader(BaseImageDatasetLoader):
    """Loader pour datasets d'images Hugging Face"""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        cache_dir: Optional[Path] = None,
        image_column: str = 'image',
        label_column: str = 'label',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir
        self.image_column = image_column
        self.label_column = label_column

    def load_data(self) -> Dataset:
        """Charge le dataset Hugging Face"""
        hf_dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            cache_dir=str(self.cache_dir) if self.cache_dir else None
        )

        return HuggingFaceImageDataset(
            hf_dataset=hf_dataset,
            transforms=self.transforms,
            image_column=self.image_column,
            label_column=self.label_column
        )
