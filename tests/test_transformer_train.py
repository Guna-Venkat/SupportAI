"""
test_transformer_train.py
=========================
Unit tests for Phase 6: DistilBERT Fine-tuning.
"""

from src.models.transformer.train import train_model
from src.utils.constants import OUTPUT_DIR


def test_train_model_smoke_pipeline(tmp_path) -> None:
    # Run mock training loop on small subset of inputs
    test_metrics = train_model(smoke_run=True, resume=False)

    assert "accuracy" in test_metrics
    assert "f1_weighted" in test_metrics

    # Check that required outputs are created
    best_model_dir = OUTPUT_DIR / "models" / "best_model"
    history_csv = OUTPUT_DIR / "metrics" / "training_history.csv"
    metrics_json = OUTPUT_DIR / "metrics" / "metrics.json"

    assert best_model_dir.exists()
    assert (best_model_dir / "config.json").exists()
    assert (best_model_dir / "model.safetensors").exists()
    assert history_csv.exists()
    assert metrics_json.exists()
