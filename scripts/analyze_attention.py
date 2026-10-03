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

SEED = 42
NUM_SAMPLES = 10
TOP_K = 5

VOCABULARY_DIR = "data/processed/vocabularies_500k"

WEIGHTS_PATH = Path(
    "models/transformer_dot_L1_seed42_500k.weights.h5"
)

OUTPUT_DIR = Path("results/attention_analysis")
OUTPUT_FILE = OUTPUT_DIR / "top_attention_samples.csv"


# ============================================================
# Model
# ============================================================

def build_model(feature_builder):
    """
    Build the exact Transformer architecture used during training.
    """

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
# Attention helper
# ============================================================

def get_top_attention(history, weights, top_k):
    """
    Return the top-k non-padding history positions.
    """

    history = np.asarray(history)
    weights = np.asarray(weights)

    valid_positions = np.where(history != 0)[0]

    if len(valid_positions) == 0:
        return []

    valid_weights = weights[valid_positions]

    order = np.argsort(valid_weights)[::-1][:top_k]

    results = []

    for idx in order:
        position = int(valid_positions[idx])
        weight = float(valid_weights[idx])

        results.append(
            {
                "position": position,
                "attention": weight,
            }
        )

    return results


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("TARGET-AWARE ATTENTION ANALYSIS")
    print("=" * 70)

    print(f"Dataset       : {DATASET_NAME}")
    print(f"Sample size   : {SAMPLE_SIZE:,}")
    print(f"Checkpoint    : {WEIGHTS_PATH}")
    print("Attention     : dot")
    print(f"Sequence len  : {MAX_SEQUENCE_LENGTH}")
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
    # Load validation split
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

    sample_df = processed_df.iloc[selected_indices].copy()

    print(
        f"Selected {len(sample_df)} validation samples "
        "for explanation."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use the exact FeatureBuilder used during training.
    # --------------------------------------------------------

    print("Loading saved vocabularies...")

    feature_builder = FeatureBuilder(
        vocabulary_dir=VOCABULARY_DIR
    )

    print(
        f"Category vocabulary : "
        f"{len(feature_builder.cate_vocab):,}"
    )

    print(
        f"Brand vocabulary    : "
        f"{len(feature_builder.brand_vocab):,}"
    )

    print(
        f"BTag vocabulary     : "
        f"{len(feature_builder.btag_vocab):,}"
    )

    print(
        f"Adgroup vocabulary  : "
        f"{len(feature_builder.adgroup_vocab):,}"
    )

    # --------------------------------------------------------
    # Convert raw IDs to model vocabulary IDs
    # --------------------------------------------------------

    print("Transforming features using saved vocabularies...")

    features, targets = feature_builder.transform(
        sample_df
    )

    print("Feature transformation complete.")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("Building Transformer...")

    model = build_model(feature_builder)

    # Create model variables
    _ = model(
        features,
        training=False,
    )

    print("Loading trained weights...")

    model.load_weights(
        WEIGHTS_PATH
    )

    print("Weights loaded successfully.")

    # --------------------------------------------------------
    # Predictions
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

    # --------------------------------------------------------
    # Attention
    # --------------------------------------------------------

    print("Extracting target-aware attention...")

    attention_weights = (
        model.explain(features)
        .numpy()
    )

    print(
        f"Attention shape: {attention_weights.shape}"
    )

    # Expected:
    #
    # (10, 50)
    #
    # 10 samples
    # 50 history positions

    # --------------------------------------------------------
    # Analyze each sample
    # --------------------------------------------------------

    rows = []

    for sample_number in range(len(sample_df)):

        row = sample_df.iloc[sample_number]

        attention = attention_weights[
            sample_number
        ]

        cate_history = np.asarray(
            row["cate_history"]
        )

        brand_history = np.asarray(
            row["brand_history"]
        )

        btag_history = np.asarray(
            row["btag_history"]
        )

        top_positions = get_top_attention(
            cate_history,
            attention,
            TOP_K,
        )

        print()
        print("-" * 70)

        print(
            f"Sample {sample_number + 1}"
        )

        print(
            f"Dataset index : "
            f"{selected_indices[sample_number]}"
        )

        print(
            f"User ID       : "
            f"{row['userid']}"
        )

        print(
            f"Actual click  : "
            f"{int(row['clk'])}"
        )

        print(
            f"Prediction    : "
            f"{predictions[sample_number]:.6f}"
        )

        print()

        print(
            "Candidate:"
        )

        print(
            f"  adgroup = {row['adgroup_id']}"
        )

        print(
            f"  category = {row['cate_id']}"
        )

        print(
            f"  brand = {row['brand']}"
        )

        print(
            f"  price = {row['price']}"
        )

        print()

        print(
            "Top attended history positions:"
        )

        for item in top_positions:

            position = item["position"]
            weight = item["attention"]

            raw_cate = cate_history[
                position
            ]

            raw_brand = brand_history[
                position
            ]

            raw_btag = btag_history[
                position
            ]

            print(
                f"  Position {position:2d} | "
                f"attention={weight:.6f} | "
                f"cate={raw_cate} | "
                f"brand={raw_brand} | "
                f"btag={raw_btag}"
            )

            rows.append(
                {
                    "sample": sample_number + 1,
                    "dataset_index": int(
                        selected_indices[
                            sample_number
                        ]
                    ),
                    "userid": row["userid"],
                    "actual_click": int(
                        row["clk"]
                    ),
                    "prediction": float(
                        predictions[
                            sample_number
                        ]
                    ),
                    "candidate_adgroup": int(
                        row["adgroup_id"]
                    ),
                    "candidate_category": int(
                        row["cate_id"]
                    ),
                    "candidate_brand": int(
                        row["brand"]
                    ),
                    "candidate_price": float(
                        row["price"]
                    ),
                    "history_position": position,
                    "attention_weight": weight,
                    "history_category": int(
                        raw_cate
                    ),
                    "history_brand": int(
                        raw_brand
                    ),
                    "history_btag": int(
                        raw_btag
                    ),
                }
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(rows)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print()
    print(
        "Existing Transformer checkpoint was used."
    )

    print(
        "No model retraining was performed."
    )


if __name__ == "__main__":
    main()