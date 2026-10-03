import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from datasets import load_dataset

from src.data.preprocess import preprocess_dataframe
from src.features.feature_builder import FeatureBuilder
from models.transformer import TransformerCTRModel


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "reczoo/TaobaoAd_x1"
CACHE_DIR = "data/raw/hf_cache"

SAMPLE_SIZE = 500_000
MAX_SEQUENCE_LENGTH = 50

NUM_SAMPLES = 1000
SEED = 42

VOCABULARY_DIR = "data/processed/vocabularies_500k"

WEIGHTS_PATH = Path(
    "models/transformer_dot_L1_seed42_500k.weights.h5"
)

OUTPUT_DIR = Path("results/attention_analysis")

SUMMARY_FILE = OUTPUT_DIR / "attention_statistics.csv"
SAMPLE_FILE = OUTPUT_DIR / "attention_sample_details.csv"


# ============================================================
# Build model
# ============================================================

def build_model(feature_builder):

    model = TransformerCTRModel(
        cate_vocab_size=len(feature_builder.cate_vocab),
        brand_vocab_size=len(feature_builder.brand_vocab),
        btag_vocab_size=len(feature_builder.btag_vocab),
        adgroup_vocab_size=len(feature_builder.adgroup_vocab),

        category_embedding_dim=16,
        brand_embedding_dim=16,
        btag_embedding_dim=8,
        adgroup_embedding_dim=16,

        hidden_dim=64,
        dropout=0.3,

        num_heads=2,
        num_layers=1,
        ff_dim=128,

        max_seq_len=50,
        attention_type="dot",
    )

    return model


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("QUANTITATIVE ATTENTION ANALYSIS")
    print("=" * 70)

    print(f"Samples       : {NUM_SAMPLES}")
    print(f"Sequence      : {MAX_SEQUENCE_LENGTH}")
    print(f"Checkpoint    : {WEIGHTS_PATH}")
    print()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        cache_dir=CACHE_DIR,
    )

    dataset = dataset.select(range(SAMPLE_SIZE))

    df = dataset.to_pandas()

    print(f"Loaded rows: {len(df):,}")

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    print("Preprocessing...")

    processed_df = preprocess_dataframe(
        df,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    # --------------------------------------------------------
    # Validation split
    # --------------------------------------------------------

    validation_indices = np.load(
        "data/processed/splits/validation_indices_500k.npy"
    )

    rng = np.random.default_rng(SEED)

    selected_indices = rng.choice(
        validation_indices,
        size=min(NUM_SAMPLES, len(validation_indices)),
        replace=False,
    )

    selected_indices = np.sort(selected_indices)

    sample_df = processed_df.iloc[
        selected_indices
    ].copy()

    print(
        f"Selected {len(sample_df):,} validation samples."
    )

    # --------------------------------------------------------
    # Feature builder
    # --------------------------------------------------------

    print("Loading vocabularies...")

    feature_builder = FeatureBuilder(
        vocabulary_dir=VOCABULARY_DIR
    )

    features, targets = feature_builder.transform(
        sample_df
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("Building model...")

    model = build_model(feature_builder)

    # Build model variables
    _ = model(
        features,
        training=False,
    )

    print("Loading trained weights...")

    model.load_weights(
        WEIGHTS_PATH
    )

    print("Weights loaded.")

    # --------------------------------------------------------
    # Predictions + attention
    # --------------------------------------------------------

    print("Generating predictions...")

    predictions = (
        model(
            features,
            training=False,
        )
        .numpy()
        .reshape(-1)
    )

    print("Extracting attention...")

    attention = (
        model.explain(features)
        .numpy()
    )

    print(
        f"Attention shape: {attention.shape}"
    )

    # ========================================================
    # Calculate statistics
    # ========================================================

    top1_values = []
    top5_values = []
    entropy_values = []

    category_overlap = []
    brand_overlap = []

    position_counts = np.zeros(
        MAX_SEQUENCE_LENGTH,
        dtype=np.int64,
    )

    detailed_rows = []

    for i in range(len(sample_df)):

        row = sample_df.iloc[i]

        weights = attention[i]

        cate_history = np.asarray(
            row["cate_history"]
        )

        brand_history = np.asarray(
            row["brand_history"]
        )

        # ----------------------------------------------------
        # Valid positions
        # ----------------------------------------------------

        valid = cate_history != 0

        valid_weights = weights[valid]

        if len(valid_weights) == 0:
            continue

        # ----------------------------------------------------
        # Normalize valid weights
        # ----------------------------------------------------

        weight_sum = valid_weights.sum()

        if weight_sum <= 0:
            continue

        valid_weights = (
            valid_weights / weight_sum
        )

        # ----------------------------------------------------
        # Top-1 attention
        # ----------------------------------------------------

        sorted_weights = np.sort(
            valid_weights
        )[::-1]

        top1 = sorted_weights[0]

        top5 = sorted_weights[
            :min(5, len(sorted_weights))
        ].sum()

        top1_values.append(top1)
        top5_values.append(top5)

        # ----------------------------------------------------
        # Attention entropy
        # ----------------------------------------------------

        entropy = -np.sum(
            valid_weights
            * np.log(
                valid_weights + 1e-12
            )
        )

        entropy_values.append(entropy)

        # ----------------------------------------------------
        # Most attended position
        # ----------------------------------------------------

        valid_positions = np.where(valid)[0]

        max_idx = np.argmax(
            valid_weights
        )

        top_position = int(
            valid_positions[max_idx]
        )

        position_counts[top_position] += 1

        # ----------------------------------------------------
        # Candidate/history category overlap
        # ----------------------------------------------------

        candidate_category = int(
            row["cate_id"]
        )

        history_categories = set(
            int(x)
            for x in cate_history[valid]
        )

        category_match = (
            candidate_category
            in history_categories
        )

        category_overlap.append(
            int(category_match)
        )

        # ----------------------------------------------------
        # Candidate/history brand overlap
        # ----------------------------------------------------

        candidate_brand = int(
            row["brand"]
        )

        brand_history_valid = (
            brand_history[valid]
        )

        history_brands = set(
            int(x)
            for x in brand_history_valid
        )

        brand_match = (
            candidate_brand != 0
            and candidate_brand in history_brands
        )

        brand_overlap.append(
            int(brand_match)
        )

        # ----------------------------------------------------
        # Detailed sample result
        # ----------------------------------------------------

        detailed_rows.append(
            {
                "dataset_index": int(
                    selected_indices[i]
                ),
                "userid": int(
                    row["userid"]
                ),
                "actual_click": int(
                    row["clk"]
                ),
                "prediction": float(
                    predictions[i]
                ),
                "candidate_category": candidate_category,
                "candidate_brand": candidate_brand,
                "top_attention": float(top1),
                "top5_attention": float(top5),
                "attention_entropy": float(
                    entropy
                ),
                "top_attention_position": top_position,
                "category_in_history": int(
                    category_match
                ),
                "brand_in_history": int(
                    brand_match
                ),
            }
        )

    # ========================================================
    # Aggregate statistics
    # ========================================================

    mean_top1 = np.mean(
        top1_values
    )

    mean_top5 = np.mean(
        top5_values
    )

    mean_entropy = np.mean(
        entropy_values
    )

    category_overlap_rate = (
        np.mean(category_overlap)
        * 100
    )

    brand_overlap_rate = (
        np.mean(brand_overlap)
        * 100
    )

    most_attended_position = int(
        np.argmax(position_counts)
    )

    most_attended_count = int(
        position_counts[
            most_attended_position
        ]
    )

    # ========================================================
    # Print results
    # ========================================================

    print()
    print("=" * 70)
    print("ATTENTION STATISTICS")
    print("=" * 70)

    print(
        f"Samples analyzed              : "
        f"{len(top1_values):,}"
    )

    print(
        f"Mean Top-1 attention          : "
        f"{mean_top1:.6f}"
    )

    print(
        f"Mean Top-5 attention          : "
        f"{mean_top5:.6f}"
    )

    print(
        f"Mean attention entropy        : "
        f"{mean_entropy:.6f}"
    )

    print(
        f"Category appears in history   : "
        f"{category_overlap_rate:.2f}%"
    )

    print(
        f"Brand appears in history      : "
        f"{brand_overlap_rate:.2f}%"
    )

    print(
        f"Most attended position        : "
        f"{most_attended_position}"
    )

    print(
        f"Times selected as top position: "
        f"{most_attended_count}"
    )

    # ========================================================
    # Position distribution
    # ========================================================

    print()
    print("Top attention positions:")

    position_order = np.argsort(
        position_counts
    )[::-1]

    for position in position_order[:10]:

        count = position_counts[
            position
        ]

        percentage = (
            count
            / len(top1_values)
            * 100
        )

        print(
            f"  Position {position:2d}: "
            f"{count:4d} samples "
            f"({percentage:.2f}%)"
        )

    # ========================================================
    # Save summary
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.DataFrame(
        [
            {
                "samples": len(top1_values),
                "mean_top1_attention": mean_top1,
                "mean_top5_attention": mean_top5,
                "mean_attention_entropy": mean_entropy,
                "category_history_overlap_pct":
                    category_overlap_rate,
                "brand_history_overlap_pct":
                    brand_overlap_rate,
                "most_attended_position":
                    most_attended_position,
                "most_attended_position_count":
                    most_attended_count,
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    detailed = pd.DataFrame(
        detailed_rows
    )

    detailed.to_csv(
        SAMPLE_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Summary saved : {SUMMARY_FILE}"
    )

    print(
        f"Details saved : {SAMPLE_FILE}"
    )

    print()
    print(
        "No model retraining was performed."
    )


if __name__ == "__main__":
    main()