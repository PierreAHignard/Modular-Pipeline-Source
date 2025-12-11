from typing import Dict, List, Union, Any
from itertools import count
import pandas as pd
from logging import Logger
import json
from pathlib import Path


class ClassMapping:
    """
    Used for class mapping.
    """

    def __init__(
            self,
            config: Any,
            mapping: Dict[str, int] = None,
            allow_new_class_outside_preload: bool = False
    ):
        self.config = config
        self._mapping = mapping or {}
        self._allow_new_class_outside_preload = allow_new_class_outside_preload

        # Déterminer le prochain ID à partir du max existant
        start = max(self._mapping.values(), default=-1) + 1
        self._counter = count(start)

        self._size = start

    def __str__(self):
        return str(self._mapping)

    def __len__(self):
        return self._size

    def __getitem__(self,
            keys: str | pd.Series
        ):
        if isinstance(keys, pd.Series):
            return keys.map(self._get_one_item)
        elif isinstance(keys, list):
            return [self._get_one_item(x) for x in keys]
        else:
            return self._get_one_item(keys)

    def set(self, key):
        return self._get_one_item(key, True)

    def _get_one_item(self,
            key: str,
            _allow_new: bool = False
        ):
        if not isinstance(key, (str, int)):
            raise ValueError(f"Value '{key}' is not included in (str, int)")

        if key not in self._mapping:
            if _allow_new or self._allow_new_class_outside_preload:
                self._mapping[key] = next(self._counter)
                self._size += 1
            else:
                raise Exception(f"Class {key} setting outside of preload")

        return self._mapping[key]

    def preload_classes(self, classes: Union[list[str], list[int]]):
        """
        Preload classes for the mapper.

        Input:
            - classes: list[str], a list of all possible class name. Should be int or str.

        Raises:
            - ValueError, if one or more values is incorrect. All other values are still added.
        """

        problematic_keys = []
        for key in classes:
            try:
                _ = self._get_one_item(key, True)
            except ValueError:
                problematic_keys.append(key)

        if len(problematic_keys) > 0:
            raise ValueError(f"Issue with classes {problematic_keys}")

    def save(self, path: str = None):
        """
        Saves the class mapping to a JSON file.

        :param path: Optional path to save the mapping. If not set, uses config.working_directory
        """
        if path is None:
            path = self.config.working_directory / "last_classes.json"
        else:
            path = Path(path)

        # Assurer que le répertoire existe
        path.parent.mkdir(parents=True, exist_ok=True)

        # Préparer les données à sauvegarder
        data = {
            "mapping": self._mapping,
            "allow_new_class_outside_preload": self._allow_new_class_outside_preload,
            "size": self._size
        }

        # Sauvegarder en JSON
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, config: Any, path: str = None):
        """
        Loads a class mapping from a JSON file.

        :param config: Configuration object
        :param path: Optional path to load the mapping from. If not set, uses config.working_directory
        :return: ClassMapping instance with loaded data
        """
        if path is None:
            path = config.working_directory / "last_classes.json"
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Mapping file not found at {path}")

        # Charger les données depuis JSON
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Créer une nouvelle instance avec le mapping chargé
        instance = cls(
            config=config,
            mapping=data["mapping"],
            allow_new_class_outside_preload=data.get("allow_new_class_outside_preload", False)
        )

        return instance

    @property
    def name_to_idx(self):
        return self._mapping

    @property
    def idx_to_name(self):
        temp = {value: key for key, value in self._mapping.items()}

        return [temp[i] for i in range(self.size)]

    @property
    def labels(self):
        return list(self._mapping.keys())

    @property
    def mapped_labels(self):
        return list(range(0, len(self)))
