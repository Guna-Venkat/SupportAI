"""
cache.py
========
Local dataset caching utility for SupportAI.

Manages saving and loading processed Pandas DataFrames representing data
splits (train, validation, test) locally in Parquet format, bypassing redundant
network requests.
"""

from pathlib import Path

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_cache_filepath(cache_dir: Path | str, split_name: str) -> Path:
    """Helper to return path to the parquet cache file."""
    return Path(cache_dir) / f"{split_name}.parquet"


def is_cached(cache_dir: Path | str, split_name: str) -> bool:
    """Checks if a specific split is already cached locally.

    Args:
        cache_dir: Directory where cached files are stored.
        split_name: Name of the dataset split (e.g. 'train', 'val', 'test').

    Returns:
        True if the cache file exists, False otherwise.
    """
    path = _get_cache_filepath(cache_dir, split_name)
    exists = path.exists()
    logger.debug("Checking cache for %s split: %s (exists=%s)", split_name, path, exists)
    return exists


def save_to_cache(df: pd.DataFrame, cache_dir: Path | str, split_name: str) -> Path:
    """Caches a processed DataFrame split as a Parquet file.

    Args:
        df: Processed DataFrame split.
        cache_dir: Directory where cache should be written.
        split_name: Identifier name for the file.

    Returns:
        The absolute Path to the written file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = _get_cache_filepath(cache_dir, split_name)

    logger.info("Saving '%s' split to local cache: %s", split_name, target_path)
    df.to_parquet(target_path, index=False)
    return target_path


def load_from_cache(cache_dir: Path | str, split_name: str) -> pd.DataFrame:
    """Loads a cached split DataFrame from local Parquet file.

    Args:
        cache_dir: Directory where cache is stored.
        split_name: Identifier name for the file.

    Returns:
        The cached DataFrame split.

    Raises:
        FileNotFoundError: If the cache file does not exist.
    """
    target_path = _get_cache_filepath(cache_dir, split_name)
    if not target_path.exists():
        msg = f"No cache file found at: {target_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Loading '%s' split from local cache: %s", split_name, target_path)
    return pd.read_parquet(target_path)
