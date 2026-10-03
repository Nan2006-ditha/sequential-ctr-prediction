import json
from pathlib import Path

import numpy as np


class Vocabulary:
    """
    Maps raw categorical IDs to compact integer indices.

    Index 0 is reserved for padding.
    Index 1 is reserved for unknown/out-of-vocabulary values.
    """

    def __init__(self):
        self.token_to_index = {
            0: 0,
        }
        self.index_to_token = {
            0: 0,
        }

    def fit(self, values):
        """
        Build vocabulary from categorical values.
        """
        unique_values = sorted(set(int(v) for v in values if int(v) != 0))

        for raw_value in unique_values:
            if raw_value not in self.token_to_index:
                index = len(self.token_to_index)

                self.token_to_index[raw_value] = index
                self.index_to_token[index] = raw_value

    def transform(self, values):
        """
        Convert raw IDs into vocabulary indices.
        Unknown values are mapped to index 1.
        """
        return np.array(
            [
                self.token_to_index.get(int(value), 1)
                for value in values
            ],
            dtype=np.int32,
        )

    def __len__(self):
        return len(self.token_to_index)

    def save(self, path):
        """
        Save vocabulary to JSON.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.token_to_index, file)

    @classmethod
    def load(cls, path):
        """
        Load vocabulary from JSON.
        """
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        vocab = cls()

        vocab.token_to_index = {
            int(key): int(value)
            for key, value in data.items()
        }

        vocab.index_to_token = {
            value: key
            for key, value in vocab.token_to_index.items()
        }

        return vocab