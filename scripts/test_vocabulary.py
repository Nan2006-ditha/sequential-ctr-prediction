import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.vocabulary import Vocabulary

def main():
    values = [
        11175,
        4576,
        6734,
        414,
        4639,
        11175,
        4576,
    ]

    vocab = Vocabulary()

    vocab.fit(values)

    print("Vocabulary size:", len(vocab))

    print("\nRaw values:")
    print(values)

    transformed = vocab.transform(values)

    print("\nTransformed values:")
    print(transformed)

    print("\nMapping:")
    for raw_value, index in vocab.token_to_index.items():
        print(f"{raw_value} -> {index}")


if __name__ == "__main__":
    main()