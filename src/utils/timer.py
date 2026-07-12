"""
timer.py
========
Timer utility context manager for SupportAI.

Enables runtime execution tracking of code blocks or functions, printing
or writing elapsed duration to logging automatically.

Usage::

    from src.utils.timer import Timer

    with Timer("Feature Extraction"):
        features = extract_features(data)

    # Output:
    # INFO | src.utils.timer | [Timer] Task completed in 0.04s
"""

import time
from types import TracebackType
from typing import Self

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Timer:
    """A context manager to measure and log the execution time of code blocks."""

    def __init__(self, name: str, level: int = 20) -> None:
        """Initialises the timer context.

        Args:
            name: Human-readable identifier for the block being timed.
            level: Logging level (e.g., logging.INFO = 20, logging.DEBUG = 10).
        """
        self.name = name
        self.level = level
        self.start_time: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> Self:
        """Starts the timer on block entry."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Stops the timer on block exit and logs the duration."""
        if self.start_time is None:
            logger.warning("Timer '%s' exited without starting.", self.name)
            return False

        end_time = time.perf_counter()
        self.elapsed = end_time - self.start_time

        logger.log(
            self.level,
            "[Timer] %s completed in %.4f seconds",
            self.name,
            self.elapsed,
        )
        return False
