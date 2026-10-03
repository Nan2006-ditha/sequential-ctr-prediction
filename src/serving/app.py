import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer import TransformerCTRModel
from src.features.vocabulary import Vocabulary
from src.monitoring.prediction_monitor import monitor_predictions

# --------------------------------------------------
# Configuration
# --------------------------------------------------

VOCABULARY_DIR = (
    PROJECT_ROOT / "data" / "processed" / "vocabularies_500k"
)

MODEL_WEIGHTS = (
    PROJECT_ROOT
    / "models"
    / "transformer_dot_L1_seed42_500k_clean.weights.h5"
)

MAX_SEQ_LEN = 50


# --------------------------------------------------
# Load vocabularies
# --------------------------------------------------

cate_vocab = Vocabulary.load(
    VOCABULARY_DIR / "cate_id_vocab.json"
)

brand_vocab = Vocabulary.load(
    VOCABULARY_DIR / "brand_vocab.json"
)

adgroup_vocab = Vocabulary.load(
    VOCABULARY_DIR / "adgroup_id_vocab.json"
)

btag_vocab = Vocabulary.load(
    VOCABULARY_DIR / "btag_vocab.json"
)


# --------------------------------------------------
# Build model
# --------------------------------------------------

model = TransformerCTRModel(
    cate_vocab_size=len(cate_vocab),
    brand_vocab_size=len(brand_vocab),
    btag_vocab_size=len(btag_vocab),
    adgroup_vocab_size=len(adgroup_vocab),
    category_embedding_dim=16,
    brand_embedding_dim=16,
    btag_embedding_dim=8,
    adgroup_embedding_dim=16,
    hidden_dim=64,
    dropout=0.3,
    num_heads=2,
    num_layers=1,
    ff_dim=128,
    max_seq_len=MAX_SEQ_LEN,
    attention_type="dot",
)


# --------------------------------------------------
# Build model once before loading weights
# --------------------------------------------------

dummy_inputs = {
    "adgroup_id": tf.constant([1], dtype=tf.int32),
    "cate_id": tf.constant([1], dtype=tf.int32),
    "brand": tf.constant([1], dtype=tf.int32),
    "price": tf.constant([0.0], dtype=tf.float32),
    "cate_history": tf.zeros(
        (1, MAX_SEQ_LEN),
        dtype=tf.int32,
    ),
    "brand_history": tf.zeros(
        (1, MAX_SEQ_LEN),
        dtype=tf.int32,
    ),
    "btag_history": tf.zeros(
        (1, MAX_SEQ_LEN),
        dtype=tf.int32,
    ),
}

model(dummy_inputs, training=False)

model.load_weights(MODEL_WEIGHTS)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Sequential CTR Prediction API",
    description=(
        "Transformer-based sequential click-through-rate "
        "prediction service."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class PredictionRequest(BaseModel):
    adgroup_id: int
    cate_id: int
    brand: int
    price: float = Field(ge=0)

    cate_history: list[int] = Field(
        default_factory=list
    )

    brand_history: list[int] = Field(
        default_factory=list
    )

    btag_history: list[int] = Field(
        default_factory=list
    )
    @model_validator(mode="after")
    def check_history_lengths(self):
        lengths = {
            len(self.cate_history),
            len(self.brand_history),
            len(self.btag_history),
        }

        if len(lengths) > 1:
            raise ValueError(
                "cate_history, brand_history, and "
                "btag_history must be the same length"
            )

        return self

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def prepare_history(
    history: list[int],
    vocabulary: Vocabulary,
) -> list[int]:

    history = history[-MAX_SEQ_LEN:]

    transformed = vocabulary.transform(history).tolist()

    if len(transformed) < MAX_SEQ_LEN:
        transformed = (
            [0] * (MAX_SEQ_LEN - len(transformed))
            + transformed
        )

    return transformed


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "transformer_dot_L1_seed42",
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    features = {
        "adgroup_id": np.array(
            [
                adgroup_vocab.transform(
                    [request.adgroup_id]
                )[0]
            ],
            dtype=np.int32,
        ),

        "cate_id": np.array(
            [
                cate_vocab.transform(
                    [request.cate_id]
                )[0]
            ],
            dtype=np.int32,
        ),

        "brand": np.array(
            [
                brand_vocab.transform(
                    [request.brand]
                )[0]
            ],
            dtype=np.int32,
        ),

        "price": np.array(
            [request.price],
            dtype=np.float32,
        ),

        "cate_history": np.array(
            [
                prepare_history(
                    request.cate_history,
                    cate_vocab,
                )
            ],
            dtype=np.int32,
        ),

        "brand_history": np.array(
            [
                prepare_history(
                    request.brand_history,
                    brand_vocab,
                )
            ],
            dtype=np.int32,
        ),

        "btag_history": np.array(
            [
                prepare_history(
                    request.btag_history,
                    btag_vocab,
                )
            ],
            dtype=np.int32,
        ),
    }

    probability = float(
        model(features, training=False).numpy()[0][0]
    )
    monitor_predictions([probability])
    return {
        "click_probability": probability,
        "prediction": int(probability >= 0.5),
        "model": "transformer_dot_L1_seed42",
    }
