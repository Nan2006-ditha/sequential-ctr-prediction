import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from datasets import load_dataset

from src.data.preprocess import preprocess_dataframe
from src.features.feature_builder import FeatureBuilder
from src.data.dataset import create_tf_dataset


DATASET_NAME = "reczoo/TaobaoAd_x1"
CACHE_DIR = "data/raw/hf_cache"


def main():

    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        cache_dir=CACHE_DIR,
    )

    dataset = dataset.select(range(10000))

    df = dataset.to_pandas()

    print(f"Loaded {len(df):,} rows")

    # Step 1: preprocessing
    print("\nPreprocessing...")

    processed_df = preprocess_dataframe(
        df,
        max_length=50,
    )

    print("Preprocessing completed.")

    # Step 2: vocabulary transformation
    print("\nBuilding model features...")

    feature_builder = FeatureBuilder()

    features, target = feature_builder.transform(
        processed_df
    )

    print("Feature transformation completed.")

    # Step 3: TensorFlow dataset
    print("\nCreating TensorFlow dataset...")

    tf_dataset = create_tf_dataset(
        features,
        target,
        batch_size=256,
        shuffle=True,
    )

    print("TensorFlow dataset created.")

    # Inspect first batch
    batch_inputs, batch_targets = next(
        iter(tf_dataset)
    )

    print("\nFirst batch:")

    for name, value in batch_inputs.items():
        print(
            f"{name}: "
            f"shape={value.shape}, "
            f"dtype={value.dtype}"
        )

    print(
        f"target: "
        f"shape={batch_targets.shape}, "
        f"dtype={batch_targets.dtype}"
    )


if __name__ == "__main__":
    main()