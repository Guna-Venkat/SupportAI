"""
artifacts.py
============
Artifact Manager utility for SupportAI.

Provides helper methods to serialise and deserialise models, metrics, structured
tabular datasets, and visualization plots. Handles folder creation transparently.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _ensure_parent_exists(path: Path) -> None:
    """Helper to create parent directories if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_model(model: Any, path: Path | str) -> None:
    """Serialises a PyTorch or Scikit-learn model checkpoint to disk.

    Args:
        model: The model object (PyTorch module or scikit-learn estimator/pipeline).
        path: Path to target file.
    """
    path = Path(path)
    _ensure_parent_exists(path)

    logger.info("Saving model checkpoint to: %s", path)

    # Lazily import torch to keep CPU CPU-only and lightweight if scikit-learn is used.
    try:
        import torch

        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), path)
            return
    except ImportError:
        logger.debug("PyTorch is not available; falling back to joblib/pickle.")

    # Fallback to joblib for scikit-learn estimators
    import joblib

    joblib.dump(model, path)


def load_model(path: Path | str, model_class: Any | None = None) -> Any:
    """Deserialises a model checkpoint from disk.

    Args:
        path: Path to target file.
        model_class: Optional PyTorch Module instance to load state dict into.

    Returns:
        The deserialised model object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")

    logger.info("Loading model checkpoint from: %s", path)

    if model_class is not None:
        import torch

        # Load state dict into PyTorch module
        state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
        model_class.load_state_dict(state_dict)
        return model_class

    # Default loading behavior
    try:
        import torch

        # Check if it was saved as a pytorch file
        try:
            return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        except Exception as e:
            logger.debug("Failed loading model as PyTorch weights, trying joblib: %s", e)
    except ImportError:
        logger.debug("PyTorch not installed, trying joblib.")

    import joblib

    return joblib.load(path)


def save_json(data: Any, path: Path | str) -> None:
    """Saves structured Python collections to JSON.

    Args:
        data: Serializable data (dict, list, etc.).
        path: Path to target file.
    """
    path = Path(path)
    _ensure_parent_exists(path)

    logger.info("Saving JSON artifact to: %s", path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def save_metrics(metrics: dict[str, Any], path: Path | str) -> None:
    """Saves model evaluation metrics dictionary to JSON.

    Wrapper around `save_json` for explicit logging context.

    Args:
        metrics: Dictionary of metric name-value mappings.
        path: Path to target file.
    """
    logger.info("Saving model metrics...")
    save_json(metrics, path)


def save_csv(df: pd.DataFrame, path: Path | str) -> None:
    """Saves a pandas DataFrame to a tidy CSV dataset.

    Args:
        df: Pandas DataFrame object.
        path: Path to target file.
    """
    path = Path(path)
    _ensure_parent_exists(path)

    logger.info("Saving CSV dataset to: %s", path)
    df.to_csv(path, index=False, encoding="utf-8")


def save_figure(fig: Any, path: Path | str) -> None:
    """Saves a matplotlib/seaborn figure to disk.

    Args:
        fig: Matplotlib Figure or Seaborn FacetGrid.
        path: Path to target file.
    """
    path = Path(path)
    _ensure_parent_exists(path)

    logger.info("Saving visualization figure to: %s", path)
    # Ensure correct bbox settings to prevent label clipping
    fig.savefig(path, bbox_inches="tight", dpi=300)
