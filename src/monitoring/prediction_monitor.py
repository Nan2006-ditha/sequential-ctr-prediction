import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def monitor_predictions(probabilities):
    probabilities = np.asarray(probabilities, dtype=float).flatten()

    if probabilities.size == 0:
        raise ValueError("Prediction list cannot be empty")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predictions must contain finite values")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be between 0 and 1")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction_count": int(probabilities.size),
        "mean_ctr_probability": float(np.mean(probabilities)),
        "min_probability": float(np.min(probabilities)),
        "max_probability": float(np.max(probabilities)),
        "positive_prediction_rate": float(
            np.mean(probabilities >= 0.5)
        ),
    }

    output_dir = Path("logs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "prediction_monitoring.jsonl"

    with output_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(report) + "\n")

    return report


if __name__ == "__main__":
    sample_predictions = [0.03, 0.12, 0.67, 0.21, 0.81]

    result = monitor_predictions(sample_predictions)

    print(json.dumps(result, indent=4))