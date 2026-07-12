"""
logging_utils.py
================
Centralised logging setup for SupportAI.

Features
--------
- Structured format with timestamps, module names, and log levels.
- Console handler with optional colour via ``rich``.
- Rotating file handler (10 MB per file, 5 backups).
- Single ``get_logger()`` helper used by every module.
- Safe to call multiple times - handlers are not duplicated.

Usage::

    from src.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    logger.info("Training started")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5
_DEFAULT_LEVEL = logging.INFO

# Module-level guard: tracks whether root logger has been configured.
_CONFIGURED: bool = False


def setup_logging(
    level: int = _DEFAULT_LEVEL,
    log_file: Path | None = None,
    use_rich: bool = True,
) -> None:
    """Configure the root logger with console and optional file handlers.

    This function is idempotent: calling it more than once has no effect
    unless ``force=True`` is passed via ``logging.basicConfig``.

    Args:
        level: Logging level, e.g. ``logging.DEBUG``.
        log_file: Optional path for the rotating log file. The parent
            directory is created automatically if it does not exist.
        use_rich: If ``True`` and ``rich`` is installed, use
            ``rich.logging.RichHandler`` for prettier console output.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ------------------------------------------------------------------
    # Console handler
    # ------------------------------------------------------------------
    if use_rich:
        try:
            from rich.logging import RichHandler

            console_handler: logging.Handler = RichHandler(
                level=level,
                rich_tracebacks=True,
                show_path=False,
                markup=True,
            )
        except ImportError:
            console_handler = _plain_stream_handler(level, formatter)
    else:
        console_handler = _plain_stream_handler(level, formatter)

    root.addHandler(console_handler)

    # ------------------------------------------------------------------
    # Rotating file handler (optional)
    # ------------------------------------------------------------------
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers.
    for noisy in ("urllib3", "httpx", "filelock", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def _plain_stream_handler(level: int, formatter: logging.Formatter) -> logging.StreamHandler:
    """Return a plain ``StreamHandler`` writing to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def get_logger(name: str, level: int = _DEFAULT_LEVEL) -> logging.Logger:
    """Return a named logger, configuring the root logger on first call.

    Args:
        name: Typically ``__name__`` of the calling module.
        level: Minimum log level for this specific logger.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    if not _CONFIGURED:
        setup_logging(level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
