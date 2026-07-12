"""
test_logging_utils.py
=====================
Unit tests for src/utils/logging_utils.py.

Verifies:
- get_logger returns a Logger instance.
- Logger name matches the requested name.
- setup_logging is idempotent (no handler duplication).
- File handler is added when log_file is specified.
- Log file is created on disk.
"""

import logging
import tempfile
from pathlib import Path

import pytest

import src.utils.logging_utils as lu
from src.utils.logging_utils import get_logger, setup_logging


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Reset the module-level guard between tests."""
    original = lu._CONFIGURED
    lu._CONFIGURED = False
    # Clear all handlers from the root logger
    root = logging.getLogger()
    root.handlers.clear()
    yield
    # Restore after test
    lu._CONFIGURED = original
    root.handlers.clear()


class TestGetLogger:
    @pytest.mark.smoke
    def test_returns_logger_instance(self) -> None:
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches(self) -> None:
        logger = get_logger("my.custom.module")
        assert logger.name == "my.custom.module"

    def test_logger_has_correct_level(self) -> None:
        logger = get_logger("test", level=logging.DEBUG)
        assert logger.level == logging.DEBUG


class TestSetupLogging:
    @pytest.mark.smoke
    def test_idempotent_no_duplicate_handlers(self) -> None:
        """Calling setup_logging twice must not add extra handlers."""
        setup_logging(use_rich=False)
        count_after_first = len(logging.getLogger().handlers)
        setup_logging(use_rich=False)
        assert len(logging.getLogger().handlers) == count_after_first

    def test_file_handler_creates_log_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test_run.log"
        setup_logging(log_file=log_file, use_rich=False)

        logger = get_logger("test.file")
        logger.info("Hello from test")

        assert log_file.exists(), "Log file should be created on disk"

    def test_file_handler_parent_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested" / "dir" / "app.log"
        setup_logging(log_file=nested, use_rich=False)
        assert nested.parent.exists()

    def test_sets_configured_flag(self) -> None:
        assert lu._CONFIGURED is False
        setup_logging(use_rich=False)
        assert lu._CONFIGURED is True


class TestSmokeLogging:
    def test_log_message_does_not_raise(self) -> None:
        logger = get_logger("smoke.test")
        # Should not raise
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warning msg")
        logger.error("error msg")
