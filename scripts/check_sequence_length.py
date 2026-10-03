from datasets import load_dataset
from collections import Counter

DATASET_NAME = "reczoo/TaobaoAd_x1"

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

length_counts = Counter()

print("Checking sequence lengths...")

for i, row in enumerate(dataset):

    cate_history = row["cate_his"]

    if cate_history:
        length = len(cate_history.split("^"))
    else:
        length = 0

    length_counts[length] += 1

    if i >= 9999:
        break

print("\nSequence length distribution from first 10,000 rows:")

for length, count in sorted(length_counts.items()):
    print(f"Length {length}: {count}")

print("\nMinimum length:", min(length_counts))
print("Maximum length:", max(length_counts))