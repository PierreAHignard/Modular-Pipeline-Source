from typing import Dict, List
from itertools import count
import pandas as pd
from logging import Logger


class ClassMapping:
    """
    Used for class mapping.
    """

    def __init__(
            self,
            mapping: Dict[str, int] = None,
            logger: Logger = None
    ):
        self.mapping = mapping or {}
        self.logger = logger

        # Déterminer le prochain ID à partir du max existant
        start = max(self.mapping.values(), default=-1) + 1
        self._counter = count(start)
        self.size = start

    def __getitem__(self, keys: str | pd.Series):
        if isinstance(keys, pd.Series):
            return keys.map(self._get_one_item)
        elif isinstance(keys, list):
            return [self._get_one_item(x) for x in keys]
        else:
            return self._get_one_item(keys)

    def _get_one_item(self, key: str):
        if not isinstance(key, (str, int)):
            raise ValueError(f"Value '{key}' is not included in (str, int)")

        if key not in self.mapping:
            self.mapping[key] = next(self._counter)
            self.size += 1

            # TODO logging
            # if self.logger:
            # self.logger.info(f"Created class mapping '{key}' -> '{self.mapping[key]}' ")

        return self.mapping[key]

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
                _ = self._get_one_item(key)
            except ValueError:
                problematic_keys.append(key)

        if len(problematic_keys) > 0:
            raise ValueError(f"Issue with classes {problematic_keys}")

    def __str__(self):
        return self.mapping