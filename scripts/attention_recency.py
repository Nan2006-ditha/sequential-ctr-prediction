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

SUMMARY_FILE = OUTPUT_DIR / "attention_recency_summary.csv"


# ============================================================
# Build model
# ============================================================

def build_model(feature_builder):

    return TransformerCTRModel(
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


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ATTENTION VS RECENCY ANALYSIS")
    print("=" * 70)

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

    processed_df = preprocess_dataframe(
        df,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    # --------------------------------------------------------
    # Validation samples
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
        f"Selected {len(sample_df)} validation samples."
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

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

    _ = model(
        features,
        training=False,
    )

    print("Loading weights...")

    model.load_weights(
        WEIGHTS_PATH
    )

    print("Weights loaded.")

    # --------------------------------------------------------
    # Attention
    # --------------------------------------------------------

    print("Extracting attention...")

    attention = (
        model.explain(features)
        .numpy()
    )

    print(
        f"Attention shape: {attention.shape}"
    )

    # ========================================================
    # Recency buckets
    #
    # Position 49 = newest
    #
    # Recent 5:
    # positions 45-49
    #
    # Recent 10:
    # positions 40-49
    #
    # Middle:
    # positions 20-39
    #
    # Older:
    # positions 0-19
    # ========================================================

    recent_5 = []
    recent_10 = []
    middle = []
    older = []

    valid_samples = 0

    # --------------------------------------------------------
    # Position-level attention
    # --------------------------------------------------------

    position_attention = np.zeros(
        MAX_SEQUENCE_LENGTH,
        dtype=np.float64,
    )

    position_count = np.zeros(
        MAX_SEQUENCE_LENGTH,
        dtype=np.int64,
    )

    for i in range(len(sample_df)):

        row = sample_df.iloc[i]

        weights = attention[i]

        cate_history = np.asarray(
            row["cate_history"]
        )

        # Valid = non-padding history
        valid = cate_history != 0

        if not np.any(valid):
            continue

        valid_samples += 1

        # Only valid history positions
        valid_weights = weights.copy()

        # Safety: ignore padding
        valid_weights[~valid] = 0.0

        total = valid_weights.sum()

        if total <= 0:
            continue

        # Normalize
        valid_weights = (
            valid_weights / total
        )

        # ----------------------------------------------------
        # Recency groups
        # ----------------------------------------------------

        r5 = valid_weights[45:50].sum()

        r10 = valid_weights[40:50].sum()

        mid = valid_weights[20:40].sum()

        old = valid_weights[0:20].sum()

        recent_5.append(r5)
        recent_10.append(r10)
        middle.append(mid)
        older.append(old)

        # ----------------------------------------------------
        # Position statistics
        # ----------------------------------------------------

        for position in range(
            MAX_SEQUENCE_LENGTH
        ):

            if valid[position]:

                position_attention[
                    position
                ] += valid_weights[position]

                position_count[
                    position
                ] += 1

    # ========================================================
    # Aggregate
    # ========================================================

    mean_recent_5 = np.mean(
        recent_5
    )

    mean_recent_10 = np.mean(
        recent_10
    )

    mean_middle = np.mean(
        middle
    )

    mean_older = np.mean(
        older
    )

    # ========================================================
    # Print results
    # ========================================================

    print()
    print("=" * 70)
    print("RECENCY ATTENTION RESULTS")
    print("=" * 70)

    print(
        f"Valid samples: {valid_samples}"
    )

    print()

    print(
        f"Recent 5 positions  : "
        f"{mean_recent_5:.6f} "
        f"({mean_recent_5 * 100:.2f}%)"
    )

    print(
        f"Recent 10 positions : "
        f"{mean_recent_10:.6f} "
        f"({mean_recent_10 * 100:.2f}%)"
    )

    print(
        f"Middle positions    : "
        f"{mean_middle:.6f} "
        f"({mean_middle * 100:.2f}%)"
    )

    print(
        f"Older positions     : "
        f"{mean_older:.6f} "
        f"({mean_older * 100:.2f}%)"
    )

    print()

    print("=" * 70)
    print("POSITION-LEVEL ATTENTION")
    print("=" * 70)

    position_rows = []

    for position in range(
        MAX_SEQUENCE_LENGTH
    ):

        if position_count[position] == 0:
            continue

        mean_attention = (
            position_attention[position]
            / position_count[position]
        )

        position_rows.append(
            {
                "position": position,
                "mean_attention": mean_attention,
                "valid_count":
                    position_count[position],
            }
        )

    position_df = pd.DataFrame(
        position_rows
    )

    position_df = position_df.sort_values(
        "mean_attention",
        ascending=False,
    )

    print(
        position_df.head(10).to_string(
            index=False
        )
    )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.DataFrame(
        [
            {
                "valid_samples": valid_samples,

                "recent_5_attention":
                    mean_recent_5,

                "recent_10_attention":
                    mean_recent_10,

                "middle_attention":
                    mean_middle,

                "older_attention":
                    mean_older,
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    position_df.to_csv(
        OUTPUT_DIR
        / "attention_by_position.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Summary saved: {SUMMARY_FILE}"
    )

    print(
        "Position data saved: "
        "results\\attention_analysis\\"
        "attention_by_position.csv"
    )

    print()
    print(
        "No model retraining was performed."
    )


if __name__ == "__main__":
    main()