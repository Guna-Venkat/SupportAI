"""
test_timer.py
=============
Unit tests for the Timer utility (src/utils/timer.py).
"""

import logging
import time

import pytest
from src.utils.timer import Timer


def test_timer_captures_duration() -> None:
    with Timer("Test block") as t:
        time.sleep(0.05)

    assert t.elapsed is not None
    assert t.elapsed >= 0.04
    assert t.elapsed < 1.0


def test_timer_logging_handler(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that the Timer records logs using the standard logging system."""
    with caplog.at_level(logging.INFO):
        with Timer("Logged block"):
            time.sleep(0.01)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert "[Timer] Logged block completed in" in record.message
