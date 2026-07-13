"""
config.py
=========
Configuration utility for SupportAI.

Provides clean methods to load the baseline configuration (`default.yaml`)
and optionally merge it recursively with run-specific overlays (e.g.,
`train.yaml` or `inference.yaml`).

All paths in configurations are resolved relative to the base repository
directory for absolute reliability.
"""

from pathlib import Path
from typing import Any

import yaml

from src.utils.constants import CONFIG_DIR, BASE_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merges overlay dictionary overrides into base dictionary.

    Args:
        base: The target base dictionary (mutated in-place).
        overlay: The override dictionary.

    Returns:
        The merged base dictionary.
    """
    for key, value in overlay.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_config(overlay_path: Path | str | None = None) -> dict[str, Any]:
    """Loads the baseline default.yaml configuration, merging any optional overlays.

    Args:
        overlay_path: Optional path to an overlay configuration YAML file.
            If relative, resolved against `configs/`.

    Returns:
        A dictionary containing the merged configuration.
    """
    default_path = CONFIG_DIR / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default configuration file not found at: {default_path}")

    # Load baseline
    logger.debug("Loading default configuration from: %s", default_path)
    with open(default_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if overlay_path is not None:
        overlay_path = Path(overlay_path)
        # Try resolving relative to current directory first, then BASE_DIR, then CONFIG_DIR
        if overlay_path.is_absolute():
            resolved_overlay = overlay_path
        elif overlay_path.exists():
            resolved_overlay = overlay_path.resolve()
        elif (BASE_DIR / overlay_path).exists():
            resolved_overlay = (BASE_DIR / overlay_path).resolve()
        else:
            resolved_overlay = CONFIG_DIR / overlay_path

        if not resolved_overlay.exists():
            raise FileNotFoundError(f"Overlay configuration not found at: {resolved_overlay}")

        logger.info("Merging overlay configuration from: %s", resolved_overlay)
        with open(resolved_overlay, encoding="utf-8") as f:
            overlay_config = yaml.safe_load(f) or {}

        config = merge_dicts(config, overlay_config)

    return config
