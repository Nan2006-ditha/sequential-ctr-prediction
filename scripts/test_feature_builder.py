import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset

from src.features.feature_builder import FeatureBuilder
from src.data.preprocess import preprocess_dataframe

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

    print("\nRunning preprocessing...")

    processed_df = preprocess_dataframe(
        df,
        max_length=50,
    )

    print("Preprocessing completed.")

    print("\nLoading vocabularies...")

    feature_builder = FeatureBuilder()

    print("Vocabularies loaded.")

    print("\nTransforming features...")

    features, target = feature_builder.transform(
        processed_df
    )

    print("Feature transformation completed.")

    print("\nFeature shapes:")

    for name, value in features.items():
        print(f"{name}: {value.shape}")

    print(f"\nTarget shape: {target.shape}")

    print("\nTarget sample:")
    print(target[:10])

    print("\nCategory history sample:")
    print(features["cate_history"][0])

    print("\nBrand history sample:")
    print(features["brand_history"][0])

    print("\nBtag history sample:")
    print(features["btag_history"][0])


if __name__ == "__main__":
    main()