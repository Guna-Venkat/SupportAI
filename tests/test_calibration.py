"""
test_calibration.py
===================
Unit tests for Phase 8: Model Calibration.
"""

from src.evaluation.calibration import CalibrationRunner, calculate_brier_score, calculate_ece
from src.utils.constants import OUTPUT_DIR


def test_calibration_metrics() -> None:
    import numpy as np

    # Perfect prediction test case
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    labels = np.array([0, 1])

    ece = calculate_ece(probs, labels)
    brier = calculate_brier_score(probs, labels)

    assert ece >= 0.0
    assert brier >= 0.0


def test_calibration_runner_smoke_pipeline(tmp_path) -> None:
    best_model_dir = OUTPUT_DIR / "models" / "best_model"

    # Initialize calibration runner
    runner = CalibrationRunner(best_model_dir, smoke_run=True)
    metrics = runner.calibrate()

    assert "optimized_temperature" in metrics
    assert "ece_before" in metrics
    assert "ece_after" in metrics
    assert "brier_before" in metrics
    assert "brier_after" in metrics

    # Check output metrics exist
    assert (OUTPUT_DIR / "metrics" / "calibration_summary.json").exists()

    # Check output plots exist
    plots_dir = OUTPUT_DIR / "metrics" / "plots"
    assert (plots_dir / "reliability_diagram.png").exists()
    assert (plots_dir / "confidence_histogram.png").exists()
    assert (plots_dir / "threshold_study.png").exists()
