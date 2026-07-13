"""
cli.py
======
CLI commands to train and evaluate classical machine learning baselines.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import load_and_preprocess_dataset
from src.evaluation.metrics import (
    calculate_metrics,
    compute_confusion_matrix,
    generate_classification_report,
)
from src.models.baselines.pipeline import create_baseline_pipeline
from src.utils.artifacts import (
    load_model,
    save_csv,
    save_figure,
    save_metrics,
    save_model,
)
from src.utils.config import load_config
from src.utils.constants import OUTPUT_DIR
from src.utils.logging_utils import get_logger
from src.utils.timer import Timer

logger = get_logger(__name__)

MODELS = ["logistic_regression", "linear_svm", "naive_bayes"]


def train_baselines(config_overlay: Path | str | None = None) -> None:
    """Trains TF-IDF + Logistic Regression, Linear SVM, and Naive Bayes models.

    Args:
        config_overlay: Optional configuration overlay path.
    """
    logger.info("Initializing baseline training command...")
    config = load_config(config_overlay)

    # 1. Load splits
    splits = load_and_preprocess_dataset(config_overlay)
    train_df = splits["train"]

    # 2. Extract hyperparameters
    max_features = config["model"]["params"].get("max_features", 5000)
    ngram_range = tuple(config["model"]["params"].get("ngram_range", [1, 2]))
    seed = config.get("seed", 42)

    # Create models output directory
    models_dir = OUTPUT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 3. Train each model
    for model_type in MODELS:
        pipeline = create_baseline_pipeline(
            model_type=model_type,
            max_features=max_features,
            ngram_range=ngram_range,
            seed=seed,
        )

        logger.info("Fitting '%s' baseline pipeline...", model_type)
        with Timer(f"Training {model_type}") as t:
            pipeline.fit(train_df["text"], train_df["label"])

        # Record training time attribute on pipeline metadata
        pipeline.training_time_seconds = t.elapsed

        # 4. Save pipeline
        model_path = models_dir / f"{model_type}_pipeline.joblib"
        save_model(pipeline, model_path)
        logger.info("Trained model saved to: %s", model_path)


def evaluate_baselines(config_overlay: Path | str | None = None) -> None:
    """Evaluates trained baseline models on the test set and exports metrics/plots.

    Args:
        config_overlay: Optional configuration overlay path.
    """
    logger.info("Initializing baseline evaluation command...")
    load_config(config_overlay)

    # 1. Load splits
    splits = load_and_preprocess_dataset(config_overlay)
    test_df = splits["test"]

    models_dir = OUTPUT_DIR / "models"
    metrics_dir = OUTPUT_DIR / "metrics"
    figures_dir = OUTPUT_DIR / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to collect evaluation metrics
    metrics_comparison = {}

    # DataFrame to collect all predictions side-by-side
    predictions_df = test_df.copy()

    # Classification report content accumulator
    report_text_accumulator = []

    # Confusion matrix plots list
    confusion_matrices = {}

    for model_type in MODELS:
        model_path = models_dir / f"{model_type}_pipeline.joblib"
        if not model_path.exists():
            msg = f"Model artifact not found for '{model_type}' at: {model_path}. Train first!"
            logger.error(msg)
            raise FileNotFoundError(msg)

        pipeline = load_model(model_path)
        model_size = Path(model_path).stat().st_size

        # Measure prediction latency
        logger.info("Measuring latency for '%s'...", model_type)
        start_time = time.perf_counter()
        preds = pipeline.predict(test_df["text"])
        elapsed = time.perf_counter() - start_time
        latency_ms_per_sample = (elapsed / len(test_df)) * 1000

        # Save predictions
        predictions_df[f"pred_{model_type}"] = preds

        # Compute classification metrics
        y_true = test_df["label"].tolist()
        class_metrics = calculate_metrics(y_true, preds, average="weighted")

        # Get fitted label encoder target mappings
        # In this phase, we map class indices back to strings if possible
        encoder_path = OUTPUT_DIR / "models" / "label_encoder.json"
        target_names = None
        if encoder_path.exists():
            with open(encoder_path, encoding="utf-8") as f:
                mapping = json.load(f)
            # Reconstruct class names sorted by their integer ids
            id_map = {int(k): v for k, v in mapping.get("id_to_label", {}).items()}
            target_names = [id_map[i] for i in sorted(id_map.keys())]

        report_str = generate_classification_report(
            y_true, preds, target_names=target_names, output_dict=False
        )
        report_text_accumulator.append(
            f"==================================================\n"
            f"MODEL: {model_type.upper()}\n"
            f"==================================================\n"
            f"{report_str}\n\n"
        )

        # Confusion Matrix
        cm = compute_confusion_matrix(y_true, preds)
        confusion_matrices[model_type] = cm

        # Merge statistics
        metrics_comparison[model_type] = {
            "accuracy": class_metrics["accuracy"],
            "precision_weighted": class_metrics["precision_weighted"],
            "recall_weighted": class_metrics["recall_weighted"],
            "f1_weighted": class_metrics["f1_weighted"],
            "training_time_seconds": getattr(pipeline, "training_time_seconds", 0.0),
            "prediction_latency_ms_per_sample": latency_ms_per_sample,
            "model_size_bytes": model_size,
        }

    # 2. Save Metrics JSON
    metrics_path = metrics_dir / "metrics.json"
    save_metrics(metrics_comparison, metrics_path)

    # 3. Save Predictions CSV
    predictions_path = metrics_dir / "predictions.csv"
    save_csv(predictions_df, predictions_path)

    # 4. Save Classification Report Text
    report_path = metrics_dir / "classification_report.txt"
    logger.info("Saving classification report to: %s", report_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_text_accumulator)

    # 5. Save Confusion Matrix Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, model_type in enumerate(MODELS):
        cm = confusion_matrices[model_type]
        # Keep it readable: display percentage or standard count.
        # Since Banking77 has 77 classes, rendering labels is too dense.
        # We plot heatmaps without class labels for high visual overview.
        sns.heatmap(
            cm,
            ax=axes[i],
            cmap="Blues",
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        axes[i].set_title(f"{model_type.replace('_', ' ').title()}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    plt.tight_layout()
    cm_path = figures_dir / "confusion_matrix.png"
    save_figure(fig, cm_path)
    plt.close()
    logger.info("Baseline evaluation reports generated successfully.")


def train_main() -> None:
    """Entry point for train_baselines command."""
    parser = argparse.ArgumentParser(description="Train classical ML baseline models.")
    parser.add_argument("--config", type=str, default=None, help="Path to config overlay.")
    args = parser.parse_args()
    train_baselines(args.config)


def evaluate_main() -> None:
    """Entry point for evaluate_baselines command."""
    parser = argparse.ArgumentParser(description="Evaluate trained baseline models on test data.")
    parser.add_argument("--config", type=str, default=None, help="Path to config overlay.")
    args = parser.parse_args()
    evaluate_baselines(args.config)


def main() -> None:
    """Interactive command router when called directly."""
    parser = argparse.ArgumentParser(description="SupportAI Baselines Manager.")
    parser.add_argument(
        "action",
        choices=["train_baselines", "evaluate_baselines"],
        help="Command to run.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config overlay.")
    args = parser.parse_args()

    if args.action == "train_baselines":
        train_baselines(args.config)
    elif args.action == "evaluate_baselines":
        evaluate_baselines(args.config)


if __name__ == "__main__":
    main()
