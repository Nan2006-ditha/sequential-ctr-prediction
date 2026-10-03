import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from src.features.feature_builder import FeatureBuilder
from models.lstm import LSTMCTRModel


def main():

    print("Loading vocabularies...")

    feature_builder = FeatureBuilder()

    print("Creating LSTM model...")

    model = LSTMCTRModel(
        cate_vocab_size=len(feature_builder.cate_vocab),
        brand_vocab_size=len(feature_builder.brand_vocab),
        btag_vocab_size=len(feature_builder.btag_vocab),
        adgroup_vocab_size=len(feature_builder.adgroup_vocab),
        category_embedding_dim=32,
        brand_embedding_dim=32,
        btag_embedding_dim=16,
        adgroup_embedding_dim=32,
        hidden_dim=128,
        dropout=0.2,
    )

    # Build the model with one dummy batch
    dummy_inputs = {
        "adgroup_id": tf.zeros((1,), dtype=tf.int32),
        "cate_id": tf.zeros((1,), dtype=tf.int32),
        "brand": tf.zeros((1,), dtype=tf.int32),
        "cate_history": tf.zeros((1, 50), dtype=tf.int32),
        "brand_history": tf.zeros((1, 50), dtype=tf.int32),
        "btag_history": tf.zeros((1, 50), dtype=tf.int32),
        "price": tf.zeros((1,), dtype=tf.float32),
    }

    model(dummy_inputs)

    print("Model built successfully.")

    # --------------------------------------------------
    # Compile
    # --------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=5.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(
                name="auc"
            ),
            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR",
            ),
        ],
    )

    print("\nModel compiled successfully.")

    print("\nOptimizer:")
    print("Adam")

    print("\nLearning rate:")
    print("0.001")

    print("\nGradient clipping:")
    print("clipnorm = 5.0")

    print("\nLoss:")
    print("Binary Cross Entropy")

    print("\nMetrics:")
    print("AUC")
    print("PR-AUC")

    print("\nTrainable parameters:")
    print(model.count_params())


if __name__ == "__main__":
    main()