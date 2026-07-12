"""
test_data_pipeline.py
======================
Unit tests for Phase 2: Data Ingestion and Validation Pipeline.
"""

from pathlib import Path

import pandas as pd
import pytest
import torch
from src.data.cache import is_cached, load_from_cache, save_to_cache
from src.data.dataloader import create_dataloader
from src.data.dataset import stratified_split
from src.data.preprocessing import LabelEncoder, clean_text
from src.data.schema import SchemaValidationError, validate_schema
from src.data.validation import (
    check_data_leakage,
    check_label_coverage,
    compute_dataset_fingerprint,
)


# 1. Text Preprocessing Tests
def test_clean_text() -> None:
    assert clean_text("  Hello   World!\t\n") == "hello world!"
    assert clean_text("multiple     spaces") == "multiple spaces"
    assert clean_text("MIXED case Text ") == "mixed case text"
    assert clean_text(123) == "123"  # Type coercion check


# 2. Label Encoder Tests
def test_label_encoder_bidirectional() -> None:
    encoder = LabelEncoder()
    labels = ["billing", "card_lost", "billing", "refund"]
    encoder.fit(labels)

    assert encoder.transform("billing") == 0
    assert encoder.transform("card_lost") == 1
    assert encoder.transform("refund") == 2

    assert encoder.inverse_transform(0) == "billing"
    assert encoder.inverse_transform(1) == "card_lost"
    assert encoder.inverse_transform(2) == "refund"

    with pytest.raises(KeyError):
        encoder.transform("unknown_intent")

    with pytest.raises(KeyError):
        encoder.inverse_transform(99)


def test_label_encoder_serialization() -> None:
    encoder = LabelEncoder()
    labels = ["billing", "refund"]
    encoder.fit(labels)

    serialized = encoder.to_dict()
    new_encoder = LabelEncoder.from_dict(serialized)

    assert new_encoder.label_to_id == encoder.label_to_id
    assert new_encoder.id_to_label == encoder.id_to_label


# 3. Schema Validation Tests
def test_validate_schema_valid() -> None:
    df = pd.DataFrame(
        {"text": ["query one", "query two"], "label": [1, 2], "label_text": ["billing", "refund"]}
    )
    # Should not raise
    validate_schema(df)


def test_validate_schema_missing_column() -> None:
    df = pd.DataFrame(
        {
            "text": ["query one"],
            "label": [1],
            # label_text is missing
        }
    )
    with pytest.raises(SchemaValidationError, match="Missing required columns"):
        validate_schema(df)


def test_validate_schema_null_values() -> None:
    df = pd.DataFrame(
        {"text": [None, "query two"], "label": [1, 2], "label_text": ["billing", "refund"]}
    )
    with pytest.raises(SchemaValidationError, match="null value"):
        validate_schema(df)


def test_validate_schema_invalid_type() -> None:
    df = pd.DataFrame(
        {
            "text": ["query one", "query two"],
            "label": ["one", "two"],  # Not numeric/integer
            "label_text": ["billing", "refund"],
        }
    )
    with pytest.raises(SchemaValidationError, match="must contain integer values"):
        validate_schema(df)


# 4. Cache Operations Tests
def test_caching_logic(tmp_path: Path) -> None:
    df = pd.DataFrame({"text": ["query 1"], "label": [0], "label_text": ["billing"]})

    assert not is_cached(tmp_path, "train")

    save_to_cache(df, tmp_path, "train")
    assert is_cached(tmp_path, "train")

    loaded_df = load_from_cache(tmp_path, "train")
    pd.testing.assert_frame_equal(loaded_df, df)


# 5. Split and Leakage Checks
def test_data_leakage() -> None:
    train_df = pd.DataFrame({"text": ["hello", "world"]})
    val_df = pd.DataFrame({"text": ["test", "query"]})
    test_df = pd.DataFrame({"text": ["hello", "unique"]})  # "hello" overlaps with train

    leak_info = check_data_leakage(train_df, val_df, test_df)
    assert leak_info["train_test_overlap"] == 1
    assert leak_info["train_val_overlap"] == 0
    assert leak_info["val_test_overlap"] == 0


def test_label_coverage() -> None:
    train_df = pd.DataFrame({"label": [0, 1]})
    val_df = pd.DataFrame({"label": [0, 2]})  # 2 is unseen in train
    test_df = pd.DataFrame({"label": [1]})

    # Should run and log warning but not raise
    check_label_coverage(train_df, val_df, test_df)


def test_stratified_split() -> None:
    # Build large dummy dataset
    texts = [f"query_{i}" for i in range(100)]
    labels = [0] * 50 + [1] * 50
    label_texts = ["billing"] * 50 + ["refund"] * 50
    df = pd.DataFrame({"text": texts, "label": labels, "label_text": label_texts})

    train_df, val_df, test_df = stratified_split(df, 0.8, 0.1, 0.1, seed=42)

    # Total should sum up
    assert len(train_df) + len(val_df) + len(test_df) == 100
    # Each split must preserve the exact class distribution
    assert (train_df["label"] == 0).sum() == 40
    assert (train_df["label"] == 1).sum() == 40
    assert (val_df["label"] == 0).sum() == 5
    assert (val_df["label"] == 1).sum() == 5
    assert (test_df["label"] == 0).sum() == 5
    assert (test_df["label"] == 1).sum() == 5


def test_fingerprint() -> None:
    df1 = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    df2 = pd.DataFrame({"text": ["b", "a"], "label": [1, 0]})
    # Order-invariant check
    hash1 = compute_dataset_fingerprint(df1)
    hash2 = compute_dataset_fingerprint(df2)
    assert hash1 == hash2


# 6. PyTorch Dataloader wrapper test
def test_pytorch_dataloader() -> None:
    df = pd.DataFrame(
        {
            "text": ["query 1", "query 2", "query 3"],
            "label": [0, 1, 2],
            "label_text": ["billing", "refund", "card_lost"],
        }
    )

    dataloader = create_dataloader(df, batch_size=2, shuffle=False)
    batches = list(dataloader)

    assert len(batches) == 2  # batch 1: size 2, batch 2: size 1
    first_batch = batches[0]
    assert "text" in first_batch
    assert "label" in first_batch
    assert len(first_batch["text"]) == 2
    assert isinstance(first_batch["label"], torch.Tensor)
