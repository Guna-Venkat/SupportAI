"""
schema.py
=========
Data schema and validation rules for SupportAI.

Ensures incoming raw datasets and preprocessed splits comply with strict column
data types, structures, and non-null constraints before downstream consumption.
"""

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# Standard expected schema column definitions
EXPECTED_COLUMNS = {
    "text": "string",
    "label": "integer",
    "label_text": "string",
}


class SchemaValidationError(ValueError):
    """Raised when dataset features or formats violate the defined schema."""

    pass


def validate_schema(df: pd.DataFrame, dataset_name: str = "dataset") -> None:
    """Validates that a pandas DataFrame satisfies the required schema constraints.

    Checks:
    - Expected column presence.
    - Correct data type conversions/constraints.
    - Absence of missing values in essential columns (text, label).

    Args:
        df: Pandas DataFrame to validate.
        dataset_name: Display name of split for logging context.

    Raises:
        SchemaValidationError: If columns are missing, data types are invalid,
            or null values are present.
    """
    logger.debug("Validating schema compliance for: %s", dataset_name)

    # 1. Column Presence Check
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        msg = f"[{dataset_name}] Missing required columns: {missing_cols}"
        logger.error(msg)
        raise SchemaValidationError(msg)

    # 2. Missing Value Check (Nulls)
    for col in ["text", "label"]:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            msg = f"[{dataset_name}] Found {null_count} null value(s) in required column '{col}'"
            logger.error(msg)
            raise SchemaValidationError(msg)

    # 3. Data Type Verification
    # Ensure text is string-like and label is integer-like
    if not pd.api.types.is_integer_dtype(df["label"]):
        try:
            # Check if all values can be coerced to integer safely
            coerced = pd.to_numeric(df["label"], errors="raise").astype("int64")
            if not (coerced == df["label"]).all():
                raise ValueError
        except (ValueError, TypeError) as err:
            msg = f"[{dataset_name}] Column 'label' must contain integer values."
            logger.error(msg)
            raise SchemaValidationError(msg) from err

    # Ensure text column contains string types
    non_str_count = df["text"].apply(lambda val: not isinstance(val, str)).sum()
    if non_str_count > 0:
        msg = (
            f"[{dataset_name}] Column 'text' must contain string values. "
            f"Found {non_str_count} non-string row(s)."
        )
        logger.error(msg)
        raise SchemaValidationError(msg)

    logger.info("Schema validation successful for %s (%d rows)", dataset_name, len(df))
