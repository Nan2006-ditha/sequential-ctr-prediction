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

    print("Vocabularies loaded.")

    print("\nCreating LSTM model...")

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

    # --------------------------------------------------
    # Create dummy batch
    # --------------------------------------------------

    batch_size = 4
    sequence_length = 50

    inputs = {
        "adgroup_id": tf.zeros(
            (batch_size,),
            dtype=tf.int32,
        ),

        "cate_id": tf.zeros(
            (batch_size,),
            dtype=tf.int32,
        ),

        "brand": tf.zeros(
            (batch_size,),
            dtype=tf.int32,
        ),

        "cate_history": tf.zeros(
            (batch_size, sequence_length),
            dtype=tf.int32,
        ),

        "brand_history": tf.zeros(
            (batch_size, sequence_length),
            dtype=tf.int32,
        ),

        "btag_history": tf.zeros(
            (batch_size, sequence_length),
            dtype=tf.int32,
        ),

        "price": tf.zeros(
            (batch_size,),
            dtype=tf.float32,
        ),
    }

    print("\nRunning forward pass...")

    output = model(inputs)

    print("\nModel created successfully.")

    print("Output shape:")
    print(output.shape)

    print("\nPredicted click probabilities:")
    print(output.numpy())

    print("\nNumber of trainable parameters:")
    print(model.count_params())


if __name__ == "__main__":
    main()