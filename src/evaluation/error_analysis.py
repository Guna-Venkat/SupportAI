"""
error_analysis.py
=================
Systematic error analysis utilities for SupportAI.

Identifies intent classification errors, per-class accuracies, longest/shortest
failure modes, most confused intent pairs, and renders confusion heatmaps.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.artifacts import save_csv, save_figure
from src.utils.constants import OUTPUT_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ErrorAnalyzer:
    """Performs systematic diagnostic error analysis on model predictions."""

    def __init__(self, predictions_path: Path | str, model_name: str) -> None:
        """Initialises the analyzer.

        Args:
            predictions_path: Path to the predictions CSV file containing columns
                'text', 'label', 'label_text', and f'pred_{model_name}'.
            model_name: Identifier name of the model to analyze.
        """
        self.predictions_path = Path(predictions_path)
        self.model_name = model_name
        self.pred_col = f"pred_{model_name}"

        if not self.predictions_path.exists():
            msg = f"Predictions file not found at: {self.predictions_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        self.df = pd.read_csv(self.predictions_path)
        if self.pred_col not in self.df.columns:
            msg = f"Model predictions column '{self.pred_col}' not found in predictions CSV."
            logger.error(msg)
            raise KeyError(msg)

        # Standard checks and attributes setup
        self.df["is_correct"] = self.df["label"] == self.df[self.pred_col]
        self.df["text_len_chars"] = self.df["text"].apply(lambda t: len(str(t)))
        self.df["text_len_words"] = self.df["text"].apply(lambda t: len(str(t).split()))

        # Extract fitted label mappings if available
        self.encoder_path = OUTPUT_DIR / "models" / "label_encoder.json"
        self.id_to_label = {}
        if self.encoder_path.exists():
            try:
                import json

                with open(self.encoder_path, encoding="utf-8") as f:
                    mapping = json.load(f)
                self.id_to_label = {int(k): v for k, v in mapping.get("id_to_label", {}).items()}
            except Exception as e:
                logger.warning("Could not load label encoder mapping: %s", e)

        # Populate missing predicted label text if needed
        self.df["pred_label_text"] = self.df[self.pred_col].apply(
            lambda i: self.id_to_label.get(int(i), f"class_{i}")
        )

    def get_misclassified(self) -> pd.DataFrame:
        """Extracts all incorrectly classified query instances."""
        return self.df[~self.df["is_correct"]].copy()

    def get_top_mistakes(self, limit: int = 100) -> pd.DataFrame:
        """Returns the top N misclassifications."""
        # By default, we keep them as they appear in the test split
        return self.get_misclassified().head(limit)

    def get_longest_failures(self, limit: int = 10) -> pd.DataFrame:
        """Returns the longest text inputs that the model failed on."""
        return self.get_misclassified().sort_values("text_len_chars", ascending=False).head(limit)

    def get_shortest_failures(self, limit: int = 10) -> pd.DataFrame:
        """Returns the shortest text inputs that the model failed on."""
        return self.get_misclassified().sort_values("text_len_chars", ascending=True).head(limit)

    def get_per_class_metrics(self) -> pd.DataFrame:
        """Computes precision, recall, f1, support, and accuracy per class.

        Returns:
            DataFrame containing index 'class_name' and metric columns.
        """
        y_true = self.df["label"]
        y_pred = self.df[self.pred_col]

        # 1. Compute classification report dictionary
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # 2. Extract per-class data
        class_records = []
        for class_id_str, metrics in report.items():
            # Skip overall summary statistics
            if class_id_str in ["accuracy", "macro avg", "weighted avg"]:
                continue

            class_id = int(class_id_str)
            class_name = self.id_to_label.get(class_id, f"class_{class_id}")

            # Per-class accuracy: correct / total support
            class_mask = y_true == class_id
            correct = ((y_true == class_id) & (y_pred == class_id)).sum()
            total = class_mask.sum()
            class_acc = (correct / total) if total > 0 else 0.0

            class_records.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1-score"],
                    "support": metrics["support"],
                    "accuracy": class_acc,
                }
            )

        class_df = pd.DataFrame(class_records)
        return class_df.set_index("class_name")

    def get_most_confused_intents(self, limit: int = 20) -> pd.DataFrame:
        """Identifies class intent pairs that are most frequently confused.

        Returns:
            DataFrame with columns: ['true_class', 'pred_class', 'confusion_count']
        """
        misclassified = self.get_misclassified()
        confusion_counts = (
            misclassified.groupby(["label_text", "pred_label_text"])
            .size()
            .reset_index(name="confusion_count")
        )
        confused_sorted = confusion_counts.sort_values("confusion_count", ascending=False)
        return confused_sorted.head(limit)

    def plot_confusion_heatmap(self, output_path: Path | str, max_classes: int = 20) -> None:
        """Renders and saves a confusion matrix heatmap for the most confused intents.

        Args:
            output_path: Path to save the confusion matrix heatmap PNG.
            max_classes: Max unique classes to include in zoom-in heatmap.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.df["label"]
        self.df[self.pred_col]

        # 1. Identify the classes with the most errors to render a readable heatmap
        misclassified = self.get_misclassified()
        error_classes = misclassified["label_text"].value_counts().head(max_classes).index.tolist()

        if not error_classes:
            logger.warning("No misclassified cases to plot.")
            return

        # 2. Filter data for these classes
        mask = self.df["label_text"].isin(error_classes) & self.df["pred_label_text"].isin(
            error_classes
        )
        filtered_df = self.df[mask]

        if len(filtered_df) == 0:
            logger.warning("No overlap for error classes heatmap.")
            return

        # 3. Compute confusion matrix and plot
        cm = confusion_matrix(
            filtered_df["label_text"],
            filtered_df["pred_label_text"],
            labels=error_classes,
        )

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm,
            ax=ax,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=error_classes,
            yticklabels=error_classes,
        )
        plt.title(f"Confusion Matrix Heatmap (Top {max_classes} Error Classes)")
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_figure(fig, output_path)
        plt.close()
        logger.info("Confusion heatmap saved to: %s", output_path)

    def run_analysis_pipeline(self) -> dict[str, Any]:
        """Runs the complete error analysis steps and writes the CSVs and plots.

        Returns:
            Dictionary with short summary metrics.
        """
        metrics_dir = OUTPUT_DIR / "metrics"
        plots_dir = metrics_dir / "plots"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save misclassified
        misclassified_df = self.get_misclassified()
        misclassified_path = metrics_dir / "misclassified.csv"
        save_csv(misclassified_df, misclassified_path)

        # 2. Save class metrics
        class_df = self.get_per_class_metrics()
        class_path = metrics_dir / "class_metrics.csv"
        save_csv(class_df, class_path)

        # 3. Save confusion heatmap
        heatmap_path = plots_dir / "confusion_heatmap.png"
        self.plot_confusion_heatmap(heatmap_path)

        # 4. Compile summary log
        longest_fails = self.get_longest_failures(3)
        shortest_fails = self.get_shortest_failures(3)
        confused_intents = self.get_most_confused_intents(5)

        summary = {
            "model_name": self.model_name,
            "total_samples": len(self.df),
            "total_errors": len(misclassified_df),
            "error_rate": len(misclassified_df) / len(self.df),
            "longest_failures": longest_fails[["text", "label_text", "pred_label_text"]].to_dict(
                orient="records"
            ),
            "shortest_failures": shortest_fails[["text", "label_text", "pred_label_text"]].to_dict(
                orient="records"
            ),
            "top_confused_pairs": confused_intents.to_dict(orient="records"),
        }

        logger.info(
            "Error analysis completed for '%s' | Total Errors: %d",
            self.model_name,
            summary["total_errors"],
        )
        return summary


def main() -> None:
    """CLI entrypoint for error analysis execution."""
    parser = argparse.ArgumentParser(description="Systematic Error Analysis CLI.")
    parser.add_argument(
        "--predictions",
        type=str,
        default=str(OUTPUT_DIR / "metrics" / "predictions.csv"),
        help="Path to predictions CSV file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear_svm",
        help="Model name to analyze.",
    )
    args = parser.parse_args()

    try:
        analyzer = ErrorAnalyzer(args.predictions, args.model)
        summary = analyzer.run_analysis_pipeline()

        print("\n" + "=" * 50)
        print("          ERROR ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Model Name:      {summary['model_name']}")
        print(f"Total Samples:   {summary['total_samples']}")
        print(f"Total Errors:    {summary['total_errors']}")
        print(f"Error Rate:      {summary['error_rate'] * 100:.2f}%")
        print(f"Accuracy:        {(1 - summary['error_rate']) * 100:.2f}%")
        print("\nTop 3 Confused Class Pairs:")
        for record in summary["top_confused_pairs"][:3]:
            print(
                f"  True: {record['label_text']} -> Pred: {record['pred_label_text']} "
                f"(count: {record['confusion_count']})"
            )
        print("=" * 50)
        sys.exit(0)
    except Exception as e:
        logger.exception("Error analysis command execution failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
