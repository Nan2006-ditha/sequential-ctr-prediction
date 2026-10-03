from datasets import load_dataset
from collections import Counter

DATASET_NAME = "reczoo/TaobaoAd_x1"

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

counter = Counter()

for i, row in enumerate(dataset):

    counter[row["clk"]] += 1

    if i >= 99999:
        break

total = sum(counter.values())

print("Samples:", total)

for label, count in sorted(counter.items()):
    percentage = count / total * 100

    print(
        f"clk={label}: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )