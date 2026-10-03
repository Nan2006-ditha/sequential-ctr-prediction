import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.vocabulary import Vocabulary


class FeatureBuilder:
    """
    Converts preprocessed CTR data into model-ready numerical features.
    """

    def __init__(self, vocabulary_dir="data/processed/vocabularies"):
        self.vocabulary_dir = Path(vocabulary_dir)

        self.cate_vocab = Vocabulary.load(
            self.vocabulary_dir / "cate_id_vocab.json"
        )

        self.brand_vocab = Vocabulary.load(
            self.vocabulary_dir / "brand_vocab.json"
        )

        self.adgroup_vocab = Vocabulary.load(
            self.vocabulary_dir / "adgroup_id_vocab.json"
        )

        self.btag_vocab = Vocabulary.load(
            self.vocabulary_dir / "btag_vocab.json"
        )

    def transform(self, df):
        """
        Convert a preprocessed dataframe into model-ready features.
        """

        features = {}

        # Candidate advertisement features
        features["adgroup_id"] = self.adgroup_vocab.transform(
            df["adgroup_id"]
        )

        features["cate_id"] = self.cate_vocab.transform(
            df["cate_id"]
        )

        features["brand"] = self.brand_vocab.transform(
            df["brand"]
        )

        # Historical sequences
        features["cate_history"] = np.array(
            [
                self.cate_vocab.transform(history)
                for history in df["cate_history"]
            ],
            dtype=np.int32,
        )

        features["brand_history"] = np.array(
            [
                self.brand_vocab.transform(history)
                for history in df["brand_history"]
            ],
            dtype=np.int32,
        )

        features["btag_history"] = np.array(
            [
                self.btag_vocab.transform(history)
                for history in df["btag_history"]
            ],
            dtype=np.int32,
        )

        # Continuous feature
        features["price"] = df["price"].to_numpy(
            dtype=np.float32
        )

        # Target
        target = df["clk"].to_numpy(
            dtype=np.float32
        )

        return features, target