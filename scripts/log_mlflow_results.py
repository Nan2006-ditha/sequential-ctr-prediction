import json
from pathlib import Path

import mlflow
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
mlflow.set_experiment("sequential_ctr_prediction")


def load_result(filename):
    path = RESULTS_DIR / filename

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def log_seed_run(model_name, result):
    seed = result["seed"]

    with mlflow.start_run(
        run_name=f"{model_name} seed {seed}"
    ):
        # Parameters
        mlflow.log_params(
            {
                "model": result["model"],
                "seed": seed,
                "dataset": "TaobaoAd_x1",
                "sample_size": 500000,
                "sequence_length": 50,
                "batch_size": 256,
                "learning_rate": 0.0005,
                "attention": result.get("attention", "N/A"),
                "layers": result.get("layers", "N/A"),
                "heads": result.get("heads", "N/A"),
            }
        )

        # Metrics
        mlflow.log_metrics(
            {
                "auc": float(result["metrics"]["auc"]),
                "pr_auc": float(result["metrics"]["pr_auc"]),
                "val_loss": float(result["metrics"]["loss"]),
                "latency_mean_ms": float(
                    result["latency"]["mean_ms"]
                ),
                "latency_p95_ms": float(
                    result["latency"]["p95_ms"]
                ),
                "params_count": float(result["params"]),
                "weights_mb": float(result["weights_mb"]),
                "train_minutes": float(result["train_minutes"]),
                "best_epoch": float(result["best_epoch"]),
            }
        )

        # Tags
        mlflow.set_tags(
            {
                "preprocessing": "shared",
                "vocabulary": "train_only",
                "evaluation": "three_seed",
            }
        )

    print(f"Logged: {model_name} seed {seed}")


def log_aggregate_run(model_name, results):
    auc_values = [
        r["metrics"]["auc"]
        for r in results
    ]

    pr_auc_values = [
        r["metrics"]["pr_auc"]
        for r in results
    ]

    loss_values = [
        r["metrics"]["loss"]
        for r in results
    ]

    latency_mean_values = [
        r["latency"]["mean_ms"]
        for r in results
    ]

    latency_p95_values = [
        r["latency"]["p95_ms"]
        for r in results
    ]

    with mlflow.start_run(
        run_name=f"{model_name} aggregate"
    ):
        # Parameters
        mlflow.log_params(
            {
                "model": model_name,
                "dataset": "TaobaoAd_x1",
                "sample_size": 500000,
                "sequence_length": 50,
                "batch_size": 256,
                "learning_rate": 0.0005,
                "num_seeds": 3,
            }
        )

        # Aggregate metrics
        mlflow.log_metrics(
            {
                "auc_mean": float(np.mean(auc_values)),
                "auc_std": float(np.std(auc_values, ddof=1)),
                "pr_auc_mean": float(np.mean(pr_auc_values)),
                "pr_auc_std": float(np.std(pr_auc_values, ddof=1)),
                "val_loss_mean": float(np.mean(loss_values)),
                "val_loss_std": float(np.std(loss_values, ddof=1)),
                "latency_mean_ms": float(
                    np.mean(latency_mean_values)
                ),
                "latency_mean_std_ms": float(
                    np.std(latency_mean_values, ddof=1)
                ),
                "latency_p95_ms": float(
                    np.mean(latency_p95_values)
                ),
                "latency_p95_std_ms": float(
                    np.std(latency_p95_values, ddof=1)
                ),
            }
        )

        mlflow.set_tags(
            {
                "preprocessing": "shared",
                "vocabulary": "train_only",
                "evaluation": "three_seed",
                "run_type": "aggregate",
            }
        )

    print(f"Logged aggregate: {model_name}")


def main():

    lstm_files = [
        "lstm_masked_seed42.json",
        "lstm_masked_seed43.json",
        "lstm_masked_seed44.json",
    ]

    transformer_files = [
        "transformer_dot_L1_seed42.json",
        "transformer_dot_L1_seed43.json",
        "transformer_dot_L1_seed44.json",
    ]

    print("=" * 70)
    print("MLFLOW EXPERIMENT TRACKING")
    print("=" * 70)

    print("\nProcessing LSTM...")

    lstm_results = [
        load_result(filename)
        for filename in lstm_files
    ]

    for result in lstm_results:
        log_seed_run("LSTM", result)

    log_aggregate_run("LSTM", lstm_results)

    print("\nProcessing Transformer...")

    transformer_results = [
        load_result(filename)
        for filename in transformer_files
    ]

    for result in transformer_results:
        log_seed_run("Transformer", result)

    log_aggregate_run(
        "Transformer",
        transformer_results
    )

    print("\n" + "=" * 70)
    print("MLFLOW TRACKING COMPLETE")
    print("=" * 70)

    print(f"\nDatabase:")
    print(MLFLOW_DB)


if __name__ == "__main__":
    main()