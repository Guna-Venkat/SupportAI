"""
dataset.py
==========
Dataset ingestion coordinator for SupportAI.

Handles downloading the Banking77 dataset from Hugging Face, combining splits,
performing stratified splits, applying text cleaning, generating label mappings,
validating schemas, and caching splits. Exposes a CLI summary command.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Lazy-loaded inside functions to respect instructions
from src.data.cache import is_cached, load_from_cache, save_to_cache
from src.data.preprocessing import LabelEncoder, clean_text
from src.data.schema import validate_schema
from src.data.validation import run_checks_and_diagnose
from src.utils.config import load_config
from src.utils.constants import DATA_DIR, OUTPUT_DIR
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a DataFrame into Train, Val, and Test splits, stratifying by label.

    Ensures proportional class distribution across splits using only Pandas/NumPy.

    Args:
        df: Input DataFrame containing columns ['text', 'label', 'label_text'].
        train_ratio: Target proportion for training split.
        val_ratio: Target proportion for validation split.
        test_ratio: Target proportion for test split.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames.
    """
    logger.debug(
        "Performing stratified split | Ratios: (%.2f, %.2f, %.2f)",
        train_ratio,
        val_ratio,
        test_ratio,
    )

    # Normalize split ratios to sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    train_groups = []
    val_groups = []
    test_groups = []

    # Group by class label
    for _, group in df.groupby("label"):
        # Shuffle group rows deterministically
        shuffled = group.sample(frac=1.0, random_state=seed)
        n = len(shuffled)

        n_train = round(train_ratio * n)
        n_val = round(val_ratio * n)

        # Boundary condition: ensure at least 1 item in train if group has elements
        if n > 0 and n_train == 0:
            n_train = 1

        train_groups.append(shuffled.iloc[:n_train])
        val_groups.append(shuffled.iloc[n_train : n_train + n_val])
        test_groups.append(shuffled.iloc[n_train + n_val :])

    return (
        pd.concat(train_groups).sample(frac=1.0, random_state=seed).reset_index(drop=True),
        pd.concat(val_groups).sample(frac=1.0, random_state=seed).reset_index(drop=True),
        pd.concat(test_groups).sample(frac=1.0, random_state=seed).reset_index(drop=True),
    )


def load_and_preprocess_dataset(
    config_overlay: Path | str | None = None,
    force_download: bool = False,
) -> dict[str, pd.DataFrame]:
    """Orchestrates dataset download, preprocessing, validation, and caching.

    If files are cached, returns them directly unless force_download is True.

    Args:
        config_overlay: Path to an optional configuration overlay YAML.
        force_download: If True, forces redownloading from HF and re-processing.

    Returns:
        Dictionary mapping split name ('train', 'val', 'test') to cleaned DataFrame.
    """
    config = load_config(config_overlay)

    # Setup directories using resolved configs
    cache_path = DATA_DIR / config["data"].get("dataset_name", "mteb_banking77")
    seed = config.get("seed", 42)
    set_seed(seed)

    splits = ["train", "val", "test"]

    # 1. Check Cache
    if not force_download and all(is_cached(cache_path, s) for s in splits):
        logger.info("All dataset splits cached locally under: %s. Loading...", cache_path)
        return {s: load_from_cache(cache_path, s) for s in splits}

    # 2. Download from HuggingFace
    logger.info("Downloading dataset 'mteb/banking77' from HuggingFace Hub...")
    try:
        from datasets import load_dataset

        hf_dataset = load_dataset("mteb/banking77")
    except Exception as e:
        logger.error("Failed to load dataset from Hugging Face: %s", e)
        raise RuntimeError("Hugging Face Hub load error.") from e

    # 3. Combine splits to perform uniform config-based re-splitting
    train_raw = hf_dataset["train"].to_pandas()
    test_raw = hf_dataset["test"].to_pandas()
    combined = pd.concat([train_raw, test_raw]).reset_index(drop=True)

    # 4. Clean text column
    logger.info("Normalizing and cleaning text values...")
    combined["text"] = combined["text"].apply(clean_text)

    # 5. Build and Fit Label Encoder
    encoder = LabelEncoder()
    # In banking77, 'label' is integer and 'label_text' is string name.
    # Align label mapping using unique combinations
    unique_mapping = combined[["label_text", "label"]].drop_duplicates().sort_values("label")
    encoder.label_to_id = dict(
        zip(unique_mapping["label_text"], unique_mapping["label"], strict=False)
    )
    encoder.id_to_label = dict(
        zip(unique_mapping["label"], unique_mapping["label_text"], strict=False)
    )

    # Save label mapping to output dir for modeling phases
    encoder_path = OUTPUT_DIR / "models" / "label_encoder.json"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving fitted LabelEncoder mapping to: %s", encoder_path)
    with open(encoder_path, "w", encoding="utf-8") as f:
        json.dump(encoder.to_dict(), f, indent=4)

    # 6. Stratified Split
    train_df, val_df, test_df = stratified_split(
        combined,
        train_ratio=config["data"].get("train_split", 0.8),
        val_ratio=config["data"].get("val_split", 0.1),
        test_ratio=config["data"].get("test_split", 0.1),
        seed=seed,
    )

    data_splits = {"train": train_df, "val": val_df, "test": test_df}

    # 7. Validate schemas
    for name, df in data_splits.items():
        validate_schema(df, name)

    # 8. Run diagnostics and write cache
    diagnostics = run_checks_and_diagnose(train_df, val_df, test_df)
    logger.info("Dataset ingestion diagnosis: %s", json.dumps(diagnostics, indent=2))

    for name, df in data_splits.items():
        save_to_cache(df, cache_path, name)

    return data_splits


def get_dataset_summary(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Generates structured summary metrics on dataset splits."""
    summary: dict[str, Any] = {}
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # Calculate duplicate rates
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        summary[f"{name}_rows"] = len(df)
        summary[f"{name}_duplicates"] = int(df.duplicated(subset=["text"]).sum())

        # Sentence length statistics (word count based)
        lens = df["text"].apply(lambda t: len(t.split()))
        summary[f"{name}_len_min"] = int(lens.min())
        summary[f"{name}_len_max"] = int(lens.max())
        summary[f"{name}_len_mean"] = float(lens.mean())
        summary[f"{name}_len_median"] = float(lens.median())

    # Label representation
    summary["unique_intents"] = int(train_df["label"].nunique())

    # Intent class balance / frequency
    freqs = train_df["label_text"].value_counts().to_dict()
    summary["intent_frequencies"] = freqs

    # Identify rare intents (< 50 samples in training set)
    rare_intents = {k: v for k, v in freqs.items() if v < 80}
    summary["rare_intents"] = rare_intents

    return summary


def main() -> None:
    """CLI entry point to execute ingestion and print dataset details."""
    parser = argparse.ArgumentParser(description="SupportAI Ingestion CLI.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration overlay.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownloading and processing of raw datasets.",
    )

    args = parser.parse_args()

    try:
        splits = load_and_preprocess_dataset(args.config, args.force)
        summary = get_dataset_summary(splits)

        print("\n" + "=" * 50)
        print("           SUPPORTAI DATASET SUMMARY")
        print("=" * 50)
        print(f"Train rows:       {summary['train_rows']} (Dups: {summary['train_duplicates']})")
        print(f"Val rows:         {summary['val_rows']} (Dups: {summary['val_duplicates']})")
        print(f"Test rows:        {summary['test_rows']} (Dups: {summary['test_duplicates']})")
        print(f"Unique Intents:   {summary['unique_intents']}")
        print(f"Rare Intents (<80): {len(summary['rare_intents'])}")
        print(f"Avg Sentence Len (Train): {summary['train_len_mean']:.2f} words")
        print("=" * 50)
        sys.exit(0)

    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
