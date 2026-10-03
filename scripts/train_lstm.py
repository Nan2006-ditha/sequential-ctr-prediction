import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from datasets import load_dataset

from src.data.preprocess import preprocess_dataframe
from src.features.feature_builder import FeatureBuilder
from src.data.dataset import create_tf_dataset
from models.lstm import LSTMCTRModel


# ============================================================
# Configuration (identical to the Transformer run)
# ============================================================

DATASET_NAME = "reczoo/TaobaoAd_x1"
CACHE_DIR = "data/raw/hf_cache"

SAMPLE_SIZE = 500_000
MAX_SEQUENCE_LENGTH = 50

BATCH_SIZE = 256
LEARNING_RATE = 0.0005
EPOCHS = 10

TRAIN_INDICES_PATH = "data/processed/splits/train_indices_500k.npy"
VALIDATION_INDICES_PATH = "data/processed/splits/validation_indices_500k.npy"
VOCABULARY_DIR = "data/processed/vocabularies_500k"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Reproduce the original unmasked LSTM baseline.",
    )
    return parser.parse_args()


def benchmark_latency(model, features, n_runs=200, warmup=20):
    """
    Single-request inference latency (batch size 1), in milliseconds.
    Identical to the function in train_transformer_500k.py.
    """
    single = {
        name: tf.convert_to_tensor(value[:1])
        for name, value in features.items()
    }

    predict = tf.function(lambda x: model(x, training=False))

    for _ in range(warmup):
        predict(single)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict(single).numpy()
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def main():

    args = parse_args()
    use_mask = not args.no_mask
    tf.keras.utils.set_random_seed(args.seed)

    tag = f"lstm_{'masked' if use_mask else 'nomask'}_seed{args.seed}"

    # --------------------------------------------------------
    # Data (same rows, same split, same vocabularies)
    # --------------------------------------------------------

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
    print(f"Rows loaded: {len(df):,}")

    processed_df = preprocess_dataframe(
        df,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    train_indices = np.load(TRAIN_INDICES_PATH)
    validation_indices = np.load(VALIDATION_INDICES_PATH)

    train_df = processed_df.iloc[train_indices].reset_index(drop=True)
    validation_df = processed_df.iloc[validation_indices].reset_index(drop=True)

    print(f"Training rows:   {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")

    feature_builder = FeatureBuilder(vocabulary_dir=VOCABULARY_DIR)

    train_features, train_target = feature_builder.transform(train_df)
    validation_features, validation_target = feature_builder.transform(
        validation_df
    )

    train_dataset = create_tf_dataset(
        train_features,
        train_target,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_dataset = create_tf_dataset(
        validation_features,
        validation_target,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"Creating model: {tag}")
    print("=" * 60)

    model = LSTMCTRModel(
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
        use_mask=use_mask,
    )

    sample_inputs, _ = next(iter(train_dataset))
    model(sample_inputs)

    n_params = model.count_params()
    print(f"Trainable parameters: {n_params:,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=5.0,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Starting LSTM training")
    print("=" * 60)

    train_start = time.perf_counter()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    train_minutes = (time.perf_counter() - train_start) / 60

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Final validation results")
    print("=" * 60)

    results = model.evaluate(
        validation_dataset,
        return_dict=True,
        verbose=1,
    )

    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")

    latency = benchmark_latency(model, validation_features)
    print(
        f"\nLatency (batch=1): mean {latency['mean_ms']:.2f} ms, "
        f"p95 {latency['p95_ms']:.2f} ms"
    )

    # --------------------------------------------------------
    # Save weights + results
    # --------------------------------------------------------

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    weights_path = f"models/{tag}_500k.weights.h5"
    model.save_weights(weights_path)

    record = {
        "model": tag,
        "seed": args.seed,
        "use_mask": use_mask,
        "params": int(n_params),
        "weights_mb": round(os.path.getsize(weights_path) / 1e6, 2),
        "train_minutes": round(train_minutes, 2),
        "best_epoch": int(np.argmax(history.history["val_auc"]) + 1),
        "metrics": {k: float(v) for k, v in results.items()},
        "latency": latency,
    }

    results_path = f"results/{tag}.json"
    with open(results_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\nWeights saved to: {weights_path}")
    print(f"Results saved to: {results_path}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()