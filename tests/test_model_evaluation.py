"""
test_model_evaluation.py
========================
Unit tests for Phase 7: Model Evaluation.
"""

from src.evaluation.evaluation_runner import EvaluationRunner
from src.utils.constants import OUTPUT_DIR


def test_evaluation_runner_smoke_pipeline(tmp_path) -> None:
    best_model_dir = OUTPUT_DIR / "models" / "best_model"

    # Initialize evaluation runner
    runner = EvaluationRunner(best_model_dir, smoke_run=True)
    summary = runner.run_evaluation_pipeline()

    assert "accuracy" in summary
    assert "macro_avg_f1" in summary
    assert "weighted_avg_f1" in summary

    # Check output metrics exist
    assert (OUTPUT_DIR / "metrics" / "evaluation_summary.json").exists()
    assert (OUTPUT_DIR / "metrics" / "per_class_metrics.csv").exists()
    assert (OUTPUT_DIR / "metrics" / "worst_classes.csv").exists()
    assert (OUTPUT_DIR / "metrics" / "prediction_examples.csv").exists()

    # Check output plots exist
    plots_dir = OUTPUT_DIR / "metrics" / "plots"
    assert (plots_dir / "confusion_matrix.png").exists()
    assert (plots_dir / "roc_curve.png").exists()
    assert (plots_dir / "pr_curve.png").exists()
