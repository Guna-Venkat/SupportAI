"""
dataset.py
==========
Transformer-based ticket dataset with tokenizer integration and caching.
"""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TransformerTicketDataset(Dataset):
    """PyTorch Dataset that loads, tokenizes, and caches tickets for DistilBERT."""

    def __init__(
        self,
        texts: list[str] | Any,
        labels: list[int] | Any,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        cache_path: Path | str | None = None,
        use_cache: bool = True,
    ) -> None:
        """Initialises the dataset.

        Args:
            texts: List of raw query texts.
            labels: List of target integer class labels.
            model_name: Pretrained tokenizer identifier.
            max_length: Maximum padding/truncation length.
            cache_path: File path to save/load pre-tokenized cache.
            use_cache: Flag to control caching behavior.
        """
        self.texts = list(texts)
        self.labels = [int(val) for val in labels]
        self.max_length = max_length
        self.model_name = model_name
        self.use_cache = use_cache
        self.cache_path = Path(cache_path) if cache_path else None

        # Load HuggingFace tokenizer
        logger.info("Loading tokenizer: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Tokenize and cache
        self.encodings = self._get_encodings()

    def _get_encodings(self) -> dict[str, torch.Tensor]:
        """Loads encodings from cache or performs tokenization and saves."""
        if self.use_cache and self.cache_path and self.cache_path.exists():
            logger.info("Loading pre-tokenized cache from: %s", self.cache_path)
            try:
                cached_data = torch.load(self.cache_path, weights_only=True)
                # Verify length matches current texts
                if len(cached_data["input_ids"]) == len(self.texts):
                    return cached_data
                logger.warning("Cache size mismatch. Re-tokenizing...")
            except Exception as e:
                logger.warning("Failed to load cache: %s. Re-tokenizing...", e)

        logger.info("Tokenizing %d sequences...", len(self.texts))
        # Disable padding here since we will use dynamic padding in the collator!
        # Truncation must be set to handle long sentences up to max_length.
        encodings = self.tokenizer(
            self.texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # DO NOT pad yet; done dynamically in batch collator!
            return_tensors=None,  # Return Python lists for easy collator padding
        )

        # Convert to dictionary of lists/tensors
        encodings_dict = {
            "input_ids": [torch.tensor(ids) for ids in encodings["input_ids"]],
            "attention_mask": [torch.tensor(mask) for mask in encodings["attention_mask"]],
        }

        if self.use_cache and self.cache_path:
            logger.info("Saving tokenized cache to: %s", self.cache_path)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                torch.save(encodings_dict, self.cache_path)
            except Exception as e:
                logger.warning("Failed to save tokenized cache: %s", e)

        return encodings_dict

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns input_ids, attention_mask, and label for index."""
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item
