"""
validation.py
=============
Dataset integrity and split leakage validation for SupportAI.

Verifies split ratio proportions, checks label coverage, identifies duplicate
text entries, checks for data leakage (text overlap), and computes dataset
fingerprints (secure SHA-256 hashes) to ensure versioned reproducibility.
"""

import hashlib
from typing import Any

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def check_data_leakage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict[str, int]:
    """Checks for data leakage (overlapping query texts) across splits.

    Args:
        train_df: Training DataFrame split.
        val_df: Validation DataFrame split.
        test_df: Test DataFrame split.

    Returns:
        A dictionary mapping overlay types to overlap row counts.
    """
    logger.debug("Checking for text overlap leakage between splits...")

    train_set = set(train_df["text"].tolist())
    val_set = set(val_df["text"].tolist())
    test_set = set(test_df["text"].tolist())

    train_val_leak = train_set.intersection(val_set)
    train_test_leak = train_set.intersection(test_set)
    val_test_leak = val_set.intersection(test_set)

    leak_counts = {
        "train_val_overlap": len(train_val_leak),
        "train_test_overlap": len(train_test_leak),
        "val_test_overlap": len(val_test_leak),
    }

    if any(count > 0 for count in leak_counts.values()):
        logger.warning(
            "DATA LEAKAGE DETECTED! Overlaps: Train/Val: %d, Train/Test: %d, Val/Test: %d",
            leak_counts["train_val_overlap"],
            leak_counts["train_test_overlap"],
            leak_counts["val_test_overlap"],
        )
    else:
        logger.info("No data leakage detected between splits (0 overlapping query strings).")

    return leak_counts


def verify_split_proportions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    expected_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    tolerance: float = 0.05,
) -> None:
    """Verifies that split proportions roughly match expected target ratios.

    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.
        expected_ratios: Expected target ratios for (train, val, test).
        tolerance: Allowed ratio difference tolerance.
    """
    total = len(train_df) + len(val_df) + len(test_df)
    if total == 0:
        raise ValueError("Cannot verify split proportions on empty datasets.")

    actual_ratios = (len(train_df) / total, len(val_df) / total, len(test_df) / total)

    logger.info(
        "Split ratios | Expected: %s | Actual: (Train=%.2f, Val=%.2f, Test=%.2f)",
        expected_ratios,
        *actual_ratios,
    )

    for name, expected, actual in zip(
        ["train", "val", "test"], expected_ratios, actual_ratios, strict=True
    ):
        if abs(actual - expected) > tolerance:
            logger.warning(
                "Split '%s' proportion %.4f deviates from expected "
                "%.4f by more than tolerance %.4f",
                name,
                actual,
                expected,
                tolerance,
            )


def check_label_coverage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Verifies class label representation across dataset splits.

    Ensures that validation and test sets do not contain labels not seen in training.

    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.
    """
    train_labels = set(train_df["label"].unique())
    val_labels = set(val_df["label"].unique())
    test_labels = set(test_df["label"].unique())

    unseen_in_val = val_labels - train_labels
    unseen_in_test = test_labels - train_labels

    if unseen_in_val:
        logger.warning("Validation split contains labels not present in Train: %s", unseen_in_val)
    if unseen_in_test:
        logger.warning("Test split contains labels not present in Train: %s", unseen_in_test)

    logger.info(
        "Label Coverage checked | Train Unique Labels: %d | Val: %d | Test: %d",
        len(train_labels),
        len(val_labels),
        len(test_labels),
    )


def compute_dataset_fingerprint(*dfs: pd.DataFrame) -> str:
    """Computes a secure, order-invariant SHA-256 fingerprint hash of the datasets.

    Collects all texts and labels, sorts them, and hashes the contents to detect
    any modifications or variations in dataset rows.

    Args:
        dfs: Pandas DataFrames to fingerprint.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    sha = hashlib.sha256()
    # Collect values to hash
    all_rows: list[str] = []
    for df in dfs:
        # Standardise rows
        for _, row in df.iterrows():
            text_val = str(row.get("text", ""))
            label_val = str(row.get("label", ""))
            all_rows.append(f"{text_val}::{label_val}")

    # Sort to ensure order invariance
    all_rows.sort()

    for item in all_rows:
        sha.update(item.encode("utf-8"))

    fingerprint = sha.hexdigest()
    logger.debug("Computed order-invariant dataset fingerprint: %s", fingerprint)
    return fingerprint


def run_checks_and_diagnose(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict[str, Any]:
    """Runs a complete suite of diagnostic checks on the dataset splits.

    Checks:
    - Duplicates.
    - Missing values.
    - Split leakage.
    - Unique intent labels count.

    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.

    Returns:
        A dictionary containing dataset diagnostic statistics.
    """
    check_data_leakage(train_df, val_df, test_df)
    verify_split_proportions(train_df, val_df, test_df)
    check_label_coverage(train_df, val_df, test_df)

    diagnostics = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dups = int(df.duplicated(subset=["text"]).sum())
        missing = int(df.isnull().sum().sum())
        diagnostics[f"{name}_rows"] = len(df)
        diagnostics[f"{name}_duplicates"] = dups
        diagnostics[f"{name}_missing_values"] = missing
        diagnostics[f"{name}_unique_intents"] = int(df["label"].nunique())

    diagnostics["fingerprint"] = compute_dataset_fingerprint(train_df, val_df, test_df)
    return diagnostics
