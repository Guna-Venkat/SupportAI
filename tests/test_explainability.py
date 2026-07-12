"""
test_explainability.py
======================
Unit tests for Phase 10: Explainability.
"""

from src.evaluation.explainability import TicketExplainer
from src.utils.constants import OUTPUT_DIR


def test_explainability_smoke_pipeline() -> None:
    best_model_dir = OUTPUT_DIR / "models" / "best_model"

    # Initialize the explainer
    explainer = TicketExplainer(model_dir=best_model_dir)

    # Test explaining a sample text with small samples for speed
    text = "The server is unresponsive and I cannot access the database."
    result = explainer.explain_ticket(text, num_features=3, num_samples=10)

    # Verify return dictionary structure
    assert "predicted_class" in result
    assert "predicted_probability" in result
    assert "attributions" in result
    assert "explanation_html" in result

    assert isinstance(result["predicted_class"], str)
    assert 0.0 <= result["predicted_probability"] <= 1.0
    assert isinstance(result["attributions"], list)
    assert isinstance(result["explanation_html"], str)

    # Test visualization executes without error
    explainer.visualize_explanation(result, top_k=3)
