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
        else:
            return self._get_one_item(keys)

    def _get_one_item(self, key: str):
        if key in self.mapping:
            return self.mapping[key]
        else:
            self.mapping[key] = next(self._counter)
            self.size += 1

            # TODO logging
            # if self.logger:
            # self.logger.info(f"Created class mapping '{key}' -> '{self.mapping[key]}' ")

            return self.mapping[key]

    def __str__(self):
        return self.mapping