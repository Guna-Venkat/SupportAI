"""
preprocessing.py
================
Text cleaning and label encoding utilities for SupportAI.

Contains standard functions to normalize ticket query texts and build
bidirectional label mappings for categorical intent target variables.
"""

import re
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """Cleans raw text strings to normalize inputs.

    Performs:
    - Lowercasing.
    - Replacing tabs, newlines, and consecutive whitespace with a single space.
    - Stripping leading and trailing spaces.

    Args:
        text: Raw input string.

    Returns:
        Cleaned and normalized string.
    """
    if not isinstance(text, str):
        logger.warning(
            "clean_text received non-string input of type %s; coercing to string.", type(text)
        )
        text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Normalise whitespace (tabs, newlines, multiple spaces -> single space)
    text = re.sub(r"\s+", " ", text)

    # Strip boundaries
    return text.strip()


class LabelEncoder:
    """Bidirectional string-to-integer mapping for target classes."""

    def __init__(self) -> None:
        self.label_to_id: dict[str, int] = {}
        self.id_to_label: dict[int, str] = {}

    def fit(self, labels: list[str]) -> "LabelEncoder":
        """Builds mappings from a unique list of string labels.

        Args:
            labels: List of label strings.
        """
        unique_labels = sorted(set(labels))
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        self.id_to_label = dict(enumerate(unique_labels))
        logger.info(
            "LabelEncoder fitted successfully with %d unique classes.", len(self.label_to_id)
        )
        return self

    def transform(self, label: str) -> int:
        """Converts string label to integer ID.

        Args:
            label: Category string.

        Returns:
            The integer label ID.
        """
        if label not in self.label_to_id:
            raise KeyError(f"Label '{label}' was not seen during fit.")
        return self.label_to_id[label]

    def inverse_transform(self, label_id: int) -> str:
        """Converts integer ID back to category string.

        Args:
            label_id: The integer label ID.

        Returns:
            The corresponding category string.
        """
        if label_id not in self.id_to_label:
            raise KeyError(f"ID {label_id} is not mapped to any label.")
        return self.id_to_label[label_id]

    def to_dict(self) -> dict[str, Any]:
        """Serializes current mapping structures to a standard dictionary."""
        return {
            "label_to_id": self.label_to_id,
            "id_to_label": {str(k): v for k, v in self.id_to_label.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabelEncoder":
        """Instantiates and reconstructs mappings from serialized dictionary.

        Args:
            data: Serialization dict containing 'label_to_id' and 'id_to_label'.
        """
        instance = cls()
        instance.label_to_id = data.get("label_to_id", {})
        # Convert keys back to integers from string keys in JSON serialization
        raw_id_map = data.get("id_to_label", {})
        instance.id_to_label = {int(k): v for k, v in raw_id_map.items()}
        return instance
