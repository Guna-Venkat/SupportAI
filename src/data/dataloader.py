"""
dataloader.py
=============
PyTorch Dataset and DataLoader wrappers for SupportAI.

Prepares cleaned Pandas DataFrames for batch execution by wrapping them
in standard PyTorch iteration classes, enabling clean input feeds for
future neural models.
"""

from typing import Any

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TicketDataset(Dataset):
    """Custom PyTorch Dataset wrapper for customer support ticket records."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialises the dataset structure.

        Args:
            df: Cleaned DataFrame containing columns: ['text', 'label', 'label_text'].
        """
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.label_texts = df["label_text"].tolist()
        logger.debug("TicketDataset initialized with %d samples.", len(self.texts))

    def __len__(self) -> int:
        """Returns total row count."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fetches a single sample record at index.

        Args:
            idx: Row offset index.

        Returns:
            Dictionary containing 'text', 'label', and 'label_text'.
        """
        return {
            "text": self.texts[idx],
            "label": int(self.labels[idx]),
            "label_text": self.label_texts[idx],
        }


def create_dataloader(
    df: pd.DataFrame,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Helper factory to instantiate PyTorch DataLoaders.

    Args:
        df: Input Pandas DataFrame.
        batch_size: Mini-batch size.
        shuffle: If True, shuffles sample order on each epoch.
        num_workers: Number of subprocesses for data loading.

    Returns:
        Configured PyTorch DataLoader.
    """
    logger.debug(
        "Creating DataLoader | batch_size=%d | shuffle=%s | workers=%d",
        batch_size,
        shuffle,
        num_workers,
    )
    dataset = TicketDataset(df)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
