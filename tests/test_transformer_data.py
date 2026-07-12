"""
test_transformer_data.py
========================
Unit tests for Phase 5: Transformer Data Preparation.
"""

import torch
from src.models.transformer.collator import DynamicPaddingCollator
from src.models.transformer.dataset import TransformerTicketDataset


def test_transformer_ticket_dataset(tmp_path) -> None:
    texts = ["I lost my card", "how to pay a bill"]
    labels = [1, 2]
    cache_file = tmp_path / "tokenized_cache.pt"

    # 1. First load: perform tokenization and cache it
    dataset = TransformerTicketDataset(
        texts=texts,
        labels=labels,
        model_name="distilbert-base-uncased",
        max_length=64,
        cache_path=cache_file,
        use_cache=True,
    )

    assert len(dataset) == 2
    assert cache_file.exists()

    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "label" in sample
    assert sample["label"].item() == 1

    # 2. Second load: check cache hit
    dataset_cached = TransformerTicketDataset(
        texts=texts,
        labels=labels,
        model_name="distilbert-base-uncased",
        max_length=64,
        cache_path=cache_file,
        use_cache=True,
    )
    assert len(dataset_cached) == 2


def test_dynamic_padding_collator() -> None:
    # Prepare dummy samples with varying sequence lengths
    features = [
        {
            "input_ids": torch.tensor([101, 2000, 102]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "label": torch.tensor(1),
        },
        {
            "input_ids": torch.tensor([101, 2000, 3000, 4000, 102]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "label": torch.tensor(2),
        },
    ]

    # Initialize collator with pad token id 0
    collator = DynamicPaddingCollator(pad_token_id=0)
    batch = collator(features)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    # Padded sequence length must equal maximum sequence length (5)
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["labels"].shape == (2,)

    # Verify first sample padding at end
    assert batch["input_ids"][0].tolist() == [101, 2000, 102, 0, 0]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
