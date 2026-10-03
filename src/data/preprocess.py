from pathlib import Path

import numpy as np
import pandas as pd


HISTORY_COLUMNS = ["btag_his", "cate_his", "brand_his"]


def parse_history(value):
    """
    Convert a ^ separated history string into a list of integers.
    """
    if pd.isna(value) or value == "":
        return []

    return [int(x) for x in str(value).split("^")]


def pad_history(history, max_length=50, padding_value=0):
    """
    Keep the most recent max_length interactions.
    Older interactions are removed if the sequence is too long.
    Shorter sequences are left-padded with 0.
    """
    history = history[-max_length:]

    if len(history) < max_length:
        history = (
            [padding_value] * (max_length - len(history))
            + history
        )

    return history



def process_history_column(series, max_length=50):
    """
    Parse and pad an entire history column.
    """
    return np.array(
        [
            pad_history(
                parse_history(value),
                max_length
            )
            for value in series
        ],
        dtype=np.int32,
    )


def preprocess_dataframe(df, max_length=50):
    result = pd.DataFrame(index=df.index)

    # User
    result["userid"] = df["userid"].astype(np.int32)

    # Target
    result["clk"] = df["clk"].astype(np.float32)

    # Sequential histories
    result["btag_history"] = list(
        process_history_column(
            df["btag_his"],
            max_length
        )
    )

    result["cate_history"] = list(
        process_history_column(
            df["cate_his"],
            max_length
        )
    )

    result["brand_history"] = list(
        process_history_column(
            df["brand_his"],
            max_length
        )
    )

    # Candidate advertisement
    result["adgroup_id"] = df["adgroup_id"].astype(np.int32)
    result["cate_id"] = df["cate_id"].astype(np.int32)
    result["brand"] = df["brand"].astype(np.int32)
    result["price"] = df["price"].astype(np.float32)

    return result
if __name__ == "__main__":
    from datasets import load_dataset

    DATASET_NAME = "reczoo/TaobaoAd_x1"
    RAW_CACHE_DIR = "data/raw/hf_cache"

    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        cache_dir=RAW_CACHE_DIR
    )

    # Use only 10,000 rows for testing
    sample = dataset.select(range(10000))

    df = sample.to_pandas()

    print(f"Loaded {len(df):,} rows")

    # Run preprocessing
    processed_df = preprocess_dataframe(df, max_length=50)

    print("\nPreprocessing completed successfully.")

    print("\nProcessed columns:")
    print(processed_df.columns.tolist())

    print("\nProcessed shape:")
    print(processed_df.shape)

    print("\nFirst category history:")
    print(processed_df["cate_history"].iloc[0])

    print("\nFirst brand history:")
    print(processed_df["brand_history"].iloc[0])

    print("\nFirst behavior history:")
    print(processed_df["btag_history"].iloc[0])

    print("\nFirst target:")
    print(processed_df["clk"].iloc[0])