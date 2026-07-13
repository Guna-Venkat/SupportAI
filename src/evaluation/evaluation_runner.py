"""
evaluation_runner.py
===================
Evaluation suite running model inference and exporting diagnostic evaluation
artifacts including Confusion Matrices, ROC, PR Curves, and error segments.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from src.data.dataset import load_and_preprocess_dataset
from src.models.transformer.collator import DynamicPaddingCollator
from src.models.transformer.dataset import TransformerTicketDataset
from src.utils.artifacts import save_csv, save_figure, save_metrics
from src.utils.config import load_config
from src.utils.constants import OUTPUT_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class EvaluationRunner:
    """Manages model evaluation on data splits and exports diagnostic charts and tables."""

    def __init__(
        self,
        model_path: Path | str,
        config_overlay: Path | str | None = None,
        smoke_run: bool = False,
    ) -> None:
        """Initialises the runner.

        Args:
            model_path: Path to the model directory containing HF model weights.
            config_overlay: Optional config override file path.
            smoke_run: If True, executes evaluation on a tiny slice of inputs.
        """
        self.model_path = Path(model_path)
        self.config = load_config(config_overlay)
        self.smoke_run = smoke_run
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create output directories
        self.metrics_dir = OUTPUT_DIR / "metrics"
        self.plots_dir = self.metrics_dir / "plots"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load model and tokenizer
        logger.info("Loading evaluation model from: %s", self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load datasets
        splits = load_and_preprocess_dataset(config_overlay)
        self.test_df = splits["test"]

        # Load class mappings
        self.encoder_path = OUTPUT_DIR / "models" / "label_encoder.json"
        self.id_to_label = {}
        if self.encoder_path.exists():
            try:
                import json

                with open(self.encoder_path, encoding="utf-8") as f:
                    mapping = json.load(f)
                self.id_to_label = {int(k): v for k, v in mapping.get("id_to_label", {}).items()}
            except Exception as e:
                logger.warning("Could not load label mappings: %s", e)

        # Max classes determination
        self.num_classes = int(
            max(
                splits["train"]["label"].max(),
                splits["val"]["label"].max(),
                splits["test"]["label"].max(),
            )
            + 1
        )

        if self.smoke_run:
            self.test_df = self.test_df.head(16)

        self.dataset = TransformerTicketDataset(
            texts=self.test_df["text"].tolist(),
            labels=self.test_df["label"].tolist(),
            model_name=str(self.model_path),
            max_length=self.config.get("max_length", 128),
            use_cache=False,
        )

        collator = DynamicPaddingCollator(pad_token_id=self.dataset.tokenizer.pad_token_id)
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.config.get("batch_size", 16),
            shuffle=False,
            collate_fn=collator,
        )

    def run_evaluation_pipeline(self) -> dict[str, Any]:
        """Runs testing loops, computes statistics, and serializes diagnostic outputs."""
        logger.info("Starting model evaluation loop...")

        all_preds = []
        all_probs = []
        all_targets = self.dataset.labels

        with torch.no_grad():
            for batch in self.loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_probs.extend(probs.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        # 1. Classification Metrics Summary
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        summary = {
            "accuracy": report["accuracy"],
            "macro_avg_f1": report["macro avg"]["f1-score"],
            "weighted_avg_f1": report["weighted avg"]["f1-score"],
        }
        save_metrics(summary, self.metrics_dir / "evaluation_summary.json")

        # 2. Per-Class Metrics and Worst Classes
        per_class_records = []
        for class_id_str, metrics in report.items():
            if class_id_str in ["accuracy", "macro avg", "weighted avg"]:
                continue
            class_id = int(class_id_str)
            class_name = self.id_to_label.get(class_id, f"class_{class_id}")
            per_class_records.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1-score"],
                    "support": metrics["support"],
                }
            )

        per_class_df = pd.DataFrame(per_class_records)
        save_csv(per_class_df, self.metrics_dir / "per_class_metrics.csv")

        # Get Worst performing classes (lowest F1 score)
        worst_classes = per_class_df.sort_values("f1_score", ascending=True)
        save_csv(worst_classes, self.metrics_dir / "worst_classes.csv")

        # 3. Prediction Examples
        examples_df = pd.DataFrame(
            {
                "text": self.dataset.texts,
                "true_label": y_true,
                "true_label_name": [self.id_to_label.get(int(i), f"class_{i}") for i in y_true],
                "predicted_label": y_pred,
                "predicted_label_name": [
                    self.id_to_label.get(int(i), f"class_{i}") for i in y_pred
                ],
                "confidence": [probs[pred] for probs, pred in zip(y_prob, y_pred, strict=True)],
            }
        )
        save_csv(examples_df, self.metrics_dir / "prediction_examples.csv")

        # 4. Plots Generation
        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_roc_curve(y_true, y_prob)
        self._plot_pr_curve(y_true, y_prob)

        logger.info("Model evaluation pipeline executed successfully.")
        return summary

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Saves a multi-class confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        plt.tight_layout()
        save_figure(fig, self.plots_dir / "confusion_matrix.png")
        plt.close()

    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Saves a macro/micro multi-class ROC curve plot."""
        # Binarize targets for multiclass ROC calculation
        classes = np.arange(self.num_classes)
        y_true_bin = label_binarize(y_true, classes=classes)

        # Handle edge case where number of classes in binarizer doesn't match probability dims
        if y_true_bin.shape[1] != y_prob.shape[1]:
            logger.warning("ROC target mismatch. Truncating class mapping dimensions.")
            min_classes = min(y_true_bin.shape[1], y_prob.shape[1])
            y_true_bin = y_true_bin[:, :min_classes]
            y_prob = y_prob[:, :min_classes]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Compute Micro ROC curve
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(
            fpr_micro,
            tpr_micro,
            label=f"micro-average ROC curve (area = {roc_auc_micro:0.2f})",
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        ax.plot([0, 1], [0, 1], "k--", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")
        ax.grid(True)
        plt.tight_layout()
        save_figure(fig, self.plots_dir / "roc_curve.png")
        plt.close()

    def _plot_pr_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Saves a Precision-Recall curve plot."""
        classes = np.arange(self.num_classes)
        y_true_bin = label_binarize(y_true, classes=classes)

        if y_true_bin.shape[1] != y_prob.shape[1]:
            min_classes = min(y_true_bin.shape[1], y_prob.shape[1])
            y_true_bin = y_true_bin[:, :min_classes]
            y_prob = y_prob[:, :min_classes]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Compute Micro PR curve
        precision_micro, recall_micro, _ = precision_recall_curve(
            y_true_bin.ravel(), y_prob.ravel()
        )
        ax.plot(
            recall_micro,
            precision_micro,
            label="micro-average PR curve",
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        ax.grid(True)
        plt.tight_layout()
        save_figure(fig, self.plots_dir / "pr_curve.png")
        plt.close()


def main() -> None:
    """CLI entrypoint for running evaluations."""
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned transformer model.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(OUTPUT_DIR / "models" / "best_model"),
        help="Path to fine-tuned transformer model.",
    )
    parser.add_argument(
        "--smoke-run",
        action="store_true",
        help="Evaluate on a tiny subset of inputs.",
    )
    args = parser.parse_args()

    try:
        runner = EvaluationRunner(args.model_dir, smoke_run=args.smoke_run)
        runner.run_evaluation_pipeline()
    except Exception as e:
        logger.exception("Model evaluation command execution failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
