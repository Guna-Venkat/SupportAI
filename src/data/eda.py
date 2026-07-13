"""
eda.py
======
Exploratory Data Analysis (EDA) generator for SupportAI.

Computes intent frequencies, sequence length percentiles, vocabulary statistics,
identifies rare classes, and saves JSON summaries, CSV tables, and visualization
plots to the configured output folders.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Lazy imports
from src.data.dataset import load_and_preprocess_dataset
from src.utils.config import load_config
from src.utils.constants import OUTPUT_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def generate_eda_artifacts(config_overlay: Path | str | None = None) -> dict[str, Any]:
    """Generates all EDA plots, tables, and summary files for Banking77.

    Args:
        config_overlay: Path to configuration overlay.

    Returns:
        Dictionary of computed summary statistics.
    """
    logger.info("Starting EDA generation...")
    config = load_config(config_overlay)

    # 1. Load preprocessed splits
    splits = load_and_preprocess_dataset(config_overlay)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # Combine for global analysis
    full_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    # Setup directories
    eda_dir = OUTPUT_DIR / "eda"
    plots_dir = eda_dir / "plots"
    tables_dir = eda_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 2. Compute Text Length Stats
    train_lens = train_df["text"].apply(lambda t: len(t.split()))
    full_lens = full_df["text"].apply(lambda t: len(t.split()))

    # 3. Label / Intent frequencies
    train_freqs = train_df["label_text"].value_counts()
    full_freqs = full_df["label_text"].value_counts()

    # Define rare classes (classes with count < 100 in the overall dataset)
    rare_classes_threshold = 100
    rare_classes = full_freqs[full_freqs < rare_classes_threshold]

    # 4. Top Words / Vocabulary
    all_words = " ".join(train_df["text"].tolist()).split()
    word_counts = Counter(all_words)
    top_words = dict(word_counts.most_common(20))

    # 5. Build Summary Statistics Dictionary
    summary_stats = {
        "dataset_name": config["data"].get("dataset_name", "mteb/banking77"),
        "total_rows": len(full_df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "unique_classes": len(full_freqs),
        "rare_classes_count": len(rare_classes),
        "rare_classes_threshold": rare_classes_threshold,
        "vocabulary_size": len(word_counts),
        "length_stats_train": {
            "min": int(train_lens.min()),
            "max": int(train_lens.max()),
            "mean": float(train_lens.mean()),
            "median": float(train_lens.median()),
            "p95": float(train_lens.quantile(0.95)),
        },
        "length_stats_overall": {
            "min": int(full_lens.min()),
            "max": int(full_lens.max()),
            "mean": float(full_lens.mean()),
            "median": float(full_lens.median()),
            "p95": float(full_lens.quantile(0.95)),
        },
        "duplicates": {
            "train": int(train_df.duplicated(subset=["text"]).sum()),
            "val": int(val_df.duplicated(subset=["text"]).sum()),
            "test": int(test_df.duplicated(subset=["text"]).sum()),
        },
    }

    # Save summary.json
    summary_path = eda_dir / "summary.json"
    logger.info("Saving EDA summary json to: %s", summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=4)

    # 6. Generate and Save Tables (CSVs)
    # Frequency table
    freq_df = pd.DataFrame(
        {
            "intent": full_freqs.index,
            "total_count": full_freqs.values,
            "train_count": [train_freqs.get(intent, 0) for intent in full_freqs.index],
            "percentage": (full_freqs.values / len(full_df)) * 100,
        }
    )
    freq_table_path = tables_dir / "intent_frequencies.csv"
    freq_df.to_csv(freq_table_path, index=False)

    # Rare classes table
    rare_df = freq_df[freq_df["total_count"] < rare_classes_threshold].copy()
    rare_table_path = tables_dir / "rare_intents.csv"
    rare_df.to_csv(rare_table_path, index=False)

    # 7. Generate and Save Plots (PNGs)
    # A. Intent distribution (Top 25)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        y=full_freqs.index[:25],
        x=full_freqs.values[:25],
        palette="viridis",
        hue=full_freqs.index[:25],
        legend=False,
    )
    plt.title("Top 25 Intent Distribution (Banking77)")
    plt.xlabel("Number of Samples")
    plt.ylabel("Intent Label")
    intent_plot_path = plots_dir / "intent_distribution.png"
    plt.savefig(intent_plot_path, bbox_inches="tight", dpi=300)
    plt.close()

    # B. Sentence length distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(full_lens, bins=25, kde=True, color="skyblue")
    plt.title("Sentence Length Distribution (Word Count)")
    plt.xlabel("Words per Sentence")
    plt.ylabel("Frequency")
    len_plot_path = plots_dir / "sentence_lengths.png"
    plt.savefig(len_plot_path, bbox_inches="tight", dpi=300)
    plt.close()

    # C. Top words frequency
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=list(top_words.values()),
        y=list(top_words.keys()),
        palette="mako",
        hue=list(top_words.keys()),
        legend=False,
    )
    plt.title("Top 20 Most Frequent Words")
    plt.xlabel("Count")
    plt.ylabel("Word")
    words_plot_path = plots_dir / "top_words.png"
    plt.savefig(words_plot_path, bbox_inches="tight", dpi=300)
    plt.close()

    logger.info("EDA generation completed. Artifacts saved under: %s", eda_dir)
    return summary_stats


if __name__ == "__main__":
    generate_eda_artifacts()
