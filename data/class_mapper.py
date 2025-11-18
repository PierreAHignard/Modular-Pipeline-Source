from typing import Dict
from itertools import count

class ClassMapping:
    """
    Used for class mapping.
    """
    def __init__(self, mapping: Dict[str, int] = None):
        self.mapping = mapping or {}
        # Déterminer le prochain ID à partir du max existant
        start = max(self.mapping.values(), default=-1) + 1
        self._counter = count(start)

    def __getitem__(self, key: str):
        if key in self.mapping:
            return self.mapping[key]
        else:
            self.mapping[key] = next(self._counter)
            return self.mapping[key]