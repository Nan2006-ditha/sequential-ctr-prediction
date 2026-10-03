"""
Collect every results/*.json produced by the training scripts and build
the final comparison table (mean +/- std across seeds).

Usage:
    python scripts/collect_results.py
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def group_key(tag):
    """
    Examples:
        transformer_dot_L1_seed42 -> transformer_dot_L1
        lstm_masked_seed42       -> lstm_masked
    """
    return re.sub(r"_seed\d+$", "", tag)


def mean_std(values):
    values = np.asarray(values, dtype=float)

    mean = values.mean()

    std = (
        values.std(ddof=1)
        if len(values) > 1
        else float("nan")
    )

    return mean, std


def fmt(values, digits=4):
    mean, std = mean_std(values)

    if np.isnan(std):
        return f"{mean:.{digits}f}"

    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main():

    files = sorted(
        RESULTS_DIR.glob("*.json")
    )

    if not files:
        print(
            f"No result files found in {RESULTS_DIR}"
        )
        return

    # --------------------------------------------------------
    # Group result files by model
    # --------------------------------------------------------

    groups = defaultdict(list)

    for path in files:

        with open(
            path,
            encoding="utf-8",
        ) as f:

            record = json.load(f)

        groups[
            group_key(record["model"])
        ].append(record)

    # --------------------------------------------------------
    # Table columns
    # --------------------------------------------------------

    columns = [
        "Model",
        "Seeds",
        "AUC",
        "PR-AUC",
        "Val loss",
        "Latency mean (ms)",
        "Latency p95 (ms)",
        "Params",
        "Weights (MB)",
        "Train (min)",
        "Best epoch",
    ]

    rows = []

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    auc_means = {}

    for name, records in sorted(
        groups.items()
    ):

        auc_values = [
            r["metrics"]["auc"]
            for r in records
        ]

        auc_mean, auc_std = mean_std(
            auc_values
        )

        auc_means[name] = auc_mean

        rows.append(
            [
                name,

                str(len(records)),

                fmt(
                    auc_values
                ),

                fmt(
                    [
                        r["metrics"]["pr_auc"]
                        for r in records
                    ]
                ),

                fmt(
                    [
                        r["metrics"]["loss"]
                        for r in records
                    ]
                ),

                fmt(
                    [
                        r["latency"]["mean_ms"]
                        for r in records
                    ],
                    2,
                ),

                fmt(
                    [
                        r["latency"]["p95_ms"]
                        for r in records
                    ],
                    2,
                ),

                f"{int(np.mean([r['params'] for r in records])):,}",

                fmt(
                    [
                        r["weights_mb"]
                        for r in records
                    ],
                    1,
                ),

                fmt(
                    [
                        r["train_minutes"]
                        for r in records
                    ],
                    1,
                ),

                fmt(
                    [
                        r["best_epoch"]
                        for r in records
                    ],
                    1,
                ),
            ]
        )

    # --------------------------------------------------------
    # Markdown table
    # --------------------------------------------------------

    lines = [
        "| "
        + " | ".join(columns)
        + " |",
        "|"
        + "|".join(
            "---" for _ in columns
        )
        + "|",
    ]

    lines += [
        "| "
        + " | ".join(row)
        + " |"
        for row in rows
    ]

    output = "\n".join(lines)

    # --------------------------------------------------------
    # Neutral comparison summary
    # --------------------------------------------------------

    if (
        "lstm_masked" in auc_means
        and "transformer_dot_L1" in auc_means
    ):

        lstm_auc = auc_means[
            "lstm_masked"
        ]

        transformer_auc = auc_means[
            "transformer_dot_L1"
        ]

        auc_gap = abs(
            transformer_auc - lstm_auc
        )

        notes = [
            "",
            "### Comparison summary",
            "",
            (
                f"Mean AUC difference between "
                f"the two models: "
                f"{auc_gap:.4f}."
            ),
            "",
            (
                "Results are reported as "
                "mean ± sample standard deviation "
                "across three random seeds."
            ),
            "",
            (
                "The models were evaluated using "
                "the same 500K-row dataset sample, "
                "train/validation split, vocabulary, "
                "sequence length, and batch size."
            ),
        ]

        output += "\n" + "\n".join(notes)

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n"
        + output
        + "\n"
    )

    # --------------------------------------------------------
    # Save Markdown
    # --------------------------------------------------------

    comparison_md = (
        RESULTS_DIR
        / "comparison.md"
    )

    comparison_md.write_text(
        output + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    comparison_csv = (
        RESULTS_DIR
        / "comparison.csv"
    )

    with open(
        comparison_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            columns
        )

        writer.writerows(
            rows
        )

    print(
        f"Saved: {comparison_md}"
    )

    print(
        f"Saved: {comparison_csv}"
    )


if __name__ == "__main__":
    main()