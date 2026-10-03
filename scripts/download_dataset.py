from datasets import load_dataset
from pathlib import Path

DATASET_NAME = "reczoo/TaobaoAd_x1"
OUTPUT_DIR = Path("data/raw")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading dataset: {DATASET_NAME}")

dataset = load_dataset(
    DATASET_NAME,
    cache_dir=str(OUTPUT_DIR / "hf_cache")
)

print("\nDataset downloaded successfully.")
print(dataset)

for split in dataset:
    print(f"\n{split}:")
    print(dataset[split])
    print("Columns:")
    print(dataset[split].column_names)