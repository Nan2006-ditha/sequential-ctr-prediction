import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.data.preprocess import preprocess_dataframe


DATASET_NAME = "reczoo/TaobaoAd_x1"
CACHE_DIR = "data/raw/hf_cache"

SAMPLE_SIZE = 200_000


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

    print(f"Rows: {len(df):,}")

    # --------------------------------------------------
    # Target distribution
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Target distribution")
    print("=" * 60)

    click_rate = df["clk"].mean()

    print(f"Non-clicks: {(df['clk'] == 0).sum():,}")
    print(f"Clicks:     {(df['clk'] == 1).sum():,}")
    print(f"Click rate: {click_rate:.4%}")

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    processed_df = preprocess_dataframe(
        df,
        max_length=50,
    )

    print("\nPreprocessing completed.")

    # --------------------------------------------------
    # Sequence lengths
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Sequence statistics")
    print("=" * 60)

    lengths = []

    for history in processed_df["cate_history"]:

        history = np.asarray(history)

        non_padding = np.count_nonzero(history)

        lengths.append(non_padding)

    lengths = np.array(lengths)

    print(f"Minimum sequence length: {lengths.min()}")
    print(f"Maximum sequence length: {lengths.max()}")
    print(f"Mean sequence length:    {lengths.mean():.2f}")

    # --------------------------------------------------
    # Train / validation target distribution
    # --------------------------------------------------

    indices = np.arange(len(df))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=df["clk"],
    )

    train_click_rate = df.iloc[train_indices]["clk"].mean()
    val_click_rate = df.iloc[validation_indices]["clk"].mean()

    print("\n" + "=" * 60)
    print("Train / validation click rates")
    print("=" * 60)

    print(f"Training click rate:   {train_click_rate:.4%}")
    print(f"Validation click rate: {val_click_rate:.4%}")

    print("\nData check completed.")


if __name__ == "__main__":
    main()