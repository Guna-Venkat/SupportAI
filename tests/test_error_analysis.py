"""
test_error_analysis.py
======================
Unit tests for Phase 4: Systematic Error Analysis.
"""

from pathlib import Path

import pandas as pd
import pytest
from src.evaluation.error_analysis import ErrorAnalyzer
from src.utils.constants import OUTPUT_DIR


@pytest.fixture
def dummy_predictions_csv(tmp_path) -> Path:
    # Construct a dummy predictions CSV containing a mix of correct and incorrect targets.
    df = pd.DataFrame(
        {
            "text": [
                "short text",
                "this is a slightly longer query sequence",
                "extremely long query text that contains many words and character sequences",
                "another text query",
            ],
            "label": [0, 1, 2, 0],
            "label_text": ["card_lost", "billing", "refund", "card_lost"],
            "pred_linear_svm": [0, 2, 2, 1],  # Correct, Confused (1->2), Correct, Confused (0->1)
        }
    )
    csv_path = tmp_path / "dummy_predictions.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_error_analyzer_instantiation(dummy_predictions_csv) -> None:
    analyzer = ErrorAnalyzer(dummy_predictions_csv, "linear_svm")
    assert analyzer.model_name == "linear_svm"
    assert len(analyzer.df) == 4
    assert analyzer.df["is_correct"].tolist() == [True, False, True, False]


def test_error_analyzer_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        ErrorAnalyzer("non_existent_file.csv", "linear_svm")


def test_error_analyzer_missing_column_raises(dummy_predictions_csv) -> None:
    with pytest.raises(KeyError):
        ErrorAnalyzer(dummy_predictions_csv, "logistic_regression")


def test_error_analyzer_failures(dummy_predictions_csv) -> None:
    analyzer = ErrorAnalyzer(dummy_predictions_csv, "linear_svm")

    # Longest failure is index 1 ("this is a slightly longer query sequence")
    longest_fails = analyzer.get_longest_failures(limit=1)
    assert len(longest_fails) == 1
    assert longest_fails.iloc[0]["text"] == "this is a slightly longer query sequence"

    # Shortest failure is index 3 ("another text query")
    shortest_fails = analyzer.get_shortest_failures(limit=1)
    assert len(shortest_fails) == 1
    assert shortest_fails.iloc[0]["text"] == "another text query"


def test_error_analyzer_confused_intents(dummy_predictions_csv) -> None:
    analyzer = ErrorAnalyzer(dummy_predictions_csv, "linear_svm")
    confused = analyzer.get_most_confused_intents()
    assert len(confused) == 2
    # Check counts match
    assert confused.iloc[0]["confusion_count"] == 1


def test_run_analysis_pipeline_integration(dummy_predictions_csv) -> None:
    # Setup analyzer on the dummy predictions
    analyzer = ErrorAnalyzer(dummy_predictions_csv, "linear_svm")
    summary = analyzer.run_analysis_pipeline()

    assert summary["model_name"] == "linear_svm"
    assert summary["total_samples"] == 4
    assert summary["total_errors"] == 2

    # Check outputs generated
    assert (OUTPUT_DIR / "metrics" / "misclassified.csv").exists()
    assert (OUTPUT_DIR / "metrics" / "class_metrics.csv").exists()
    assert (OUTPUT_DIR / "metrics" / "plots" / "confusion_heatmap.png").exists()
