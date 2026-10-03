import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.data.preprocess import preprocess_dataframe
from src.features.vocabulary import Vocabulary


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "reczoo/TaobaoAd_x1"
CACHE_DIR = "data/raw/hf_cache"

SAMPLE_SIZE = 500_000
MAX_SEQUENCE_LENGTH = 50
RANDOM_STATE = 42

OUTPUT_DIR = Path("data/processed/vocabularies_500k")
SPLIT_DIR = Path("data/processed/splits")


# ============================================================
# Helper functions
# ============================================================

def flatten_history(series):
    values = []

    for history in series:
        values.extend(history)

    return np.array(values, dtype=np.int32)


def build_and_save_vocabulary(name, values):
    vocab = Vocabulary()
    vocab.fit(values)

    path = OUTPUT_DIR / f"{name}_vocab.json"
    vocab.save(path)

    print(f"{name} vocabulary size: {len(vocab):,}")
    print(f"Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Loading dataset")
    print("=" * 60)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        cache_dir=CACHE_DIR,
    )

    dataset = dataset.select(range(SAMPLE_SIZE))

    df = dataset.to_pandas()

    print(f"Loaded {len(df):,} rows")

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    print("\nPreprocessing...")

    processed_df = preprocess_dataframe(
        df,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    print("Preprocessing completed.")

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    print("\nCreating train/validation split...")

    indices = np.arange(len(processed_df))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=processed_df["clk"],
    )

    print(f"Training rows:   {len(train_indices):,}")
    print(f"Validation rows: {len(validation_indices):,}")

    # --------------------------------------------------------
    # Save exact split
    # --------------------------------------------------------

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        SPLIT_DIR / "train_indices_500k.npy",
        train_indices,
    )

    np.save(
        SPLIT_DIR / "validation_indices_500k.npy",
        validation_indices,
    )

    print("Saved 500K train/validation indices.")

    train_df = processed_df.iloc[train_indices]

    # --------------------------------------------------------
    # Create vocabulary directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Category vocabulary
    # Candidate + history
    # --------------------------------------------------------

    print("\nBuilding category vocabulary...")

    cate_history = flatten_history(
        train_df["cate_history"]
    )

    cate_values = np.concatenate(
        [
            train_df["cate_id"].to_numpy(),
            cate_history,
        ]
    )

    build_and_save_vocabulary(
        "cate_id",
        cate_values,
    )

    # --------------------------------------------------------
    # Brand vocabulary
    # Candidate + history
    # --------------------------------------------------------

    print("\nBuilding brand vocabulary...")

    brand_history = flatten_history(
        train_df["brand_history"]
    )

    brand_values = np.concatenate(
        [
            train_df["brand"].to_numpy(),
            brand_history,
        ]
    )

    build_and_save_vocabulary(
        "brand",
        brand_values,
    )

    # --------------------------------------------------------
    # Adgroup vocabulary
    # Candidate only
    # --------------------------------------------------------

    print("\nBuilding adgroup vocabulary...")

    build_and_save_vocabulary(
        "adgroup_id",
        train_df["adgroup_id"].to_numpy(),
    )

    # --------------------------------------------------------
    # Behavior tag vocabulary
    # History
    # --------------------------------------------------------

    print("\nBuilding btag vocabulary...")

    btag_history = flatten_history(
        train_df["btag_history"]
    )

    build_and_save_vocabulary(
        "btag",
        btag_history,
    )

    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("500K Vocabulary preparation completed")
    print("=" * 60)

    print(f"Vocabulary directory: {OUTPUT_DIR}")
    print("Train split:          data/processed/splits/train_indices_500k.npy")
    print("Validation split:     data/processed/splits/validation_indices_500k.npy")


if __name__ == "__main__":
    main()