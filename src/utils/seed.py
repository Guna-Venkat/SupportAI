"""
seed.py
=======
Reproducibility helpers for SupportAI.

Sets seeds for Python's built-in ``random``, ``numpy``, ``torch`` (CPU and
CUDA), and optionally disables CuDNN non-determinism for full
reproducibility at the cost of some performance.

Usage::

    from src.utils.seed import set_seed

    set_seed(42)                    # default – fast, deterministic enough
    set_seed(42, full_determinism=True)  # strict – slower, fully reproducible
"""

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)

# Default seed used across all experiments unless overridden via config.
DEFAULT_SEED: int = 42


def set_seed(seed: int = DEFAULT_SEED, full_determinism: bool = False) -> None:
    """Set all global random seeds for reproducibility.

    Args:
        seed: Integer seed value. Must be in ``[0, 2**32 - 1]``.
        full_determinism: If ``True``, forces CuDNN into deterministic mode
            and disables benchmark mode. This eliminates non-determinism from
            GPU kernels but may significantly reduce training throughput.
            Set to ``False`` (default) for training runs; ``True`` only when
            exact reproducibility of GPU outputs is required.

    Note:
        Even with ``full_determinism=True``, perfect bit-for-bit
        reproducibility is not guaranteed across different hardware, drivers,
        or CUDA versions.
    """
    if not (0 <= seed < 2**32):
        raise ValueError(f"Seed must be in [0, 2**32 - 1], got {seed}.")

    # Python built-in
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Torch (imported lazily to avoid mandatory GPU dependency at import time)
    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # multi-GPU

        if full_determinism:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Required for some ops to use deterministic algorithms.
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        else:
            torch.backends.cudnn.benchmark = True

    except ImportError:
        logger.warning(
            "torch is not installed; torch seeds were not set. "
            "Install torch to enable GPU reproducibility."
        )

    # Hash seed for Python's hash randomisation (affects dict / set ordering
    # in some edge cases).
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.debug(
        "Seeds set | seed=%d | full_determinism=%s",
        seed,
        full_determinism,
    )
