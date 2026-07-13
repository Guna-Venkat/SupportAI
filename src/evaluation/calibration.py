"""
calibration.py
==============
Implements Temperature Scaling calibration, Expected Calibration Error (ECE)
calculations, Brier Scores, and diagnostic plotting comparisons.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from src.data.dataset import load_and_preprocess_dataset
from src.models.transformer.collator import DynamicPaddingCollator
from src.models.transformer.dataset import TransformerTicketDataset
from src.utils.artifacts import save_figure, save_metrics
from src.utils.config import load_config
from src.utils.constants import OUTPUT_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TemperatureScaler(nn.Module):
    """Optimises a temperature parameter T to calibrate logits probabilities."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Prevent division by zero or negative temperatures
        temp = torch.clamp(self.temperature, min=1e-4)
        return logits / temp


def calculate_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """Calculates the Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find elements in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)

    return float(ece)


def calculate_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Calculates the multi-class Brier score."""
    num_classes = probs.shape[1]
    # One-hot encode targets
    labels_one_hot = np.eye(num_classes)[labels]
    # Compute mean squared difference
    brier = np.mean(np.sum((probs - labels_one_hot) ** 2, axis=1))
    return float(brier)


class CalibrationRunner:
    """Manages the calibration workflow: learning T, evaluating ECE/Brier, and plotting."""

    def __init__(
        self,
        model_path: Path | str,
        config_overlay: Path | str | None = None,
        smoke_run: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.config = load_config(config_overlay)
        self.smoke_run = smoke_run
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.metrics_dir = OUTPUT_DIR / "metrics"
        self.plots_dir = self.metrics_dir / "plots"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading model for calibration from: %s", self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        splits = load_and_preprocess_dataset(config_overlay)
        self.val_df = splits["val"]
        self.test_df = splits["test"]

        if self.smoke_run:
            self.val_df = self.val_df.head(16)
            self.test_df = self.test_df.head(16)

        self.max_length = self.config.get("max_length", 128)
        self.batch_size = self.config.get("batch_size", 16)

    def extract_logits(self, df: Any) -> tuple[np.ndarray, np.ndarray]:
        """Runs inference over a split to collect raw logits and labels."""
        dataset = TransformerTicketDataset(
            texts=df["text"].tolist(),
            labels=df["label"].tolist(),
            model_name=str(self.model_path),
            max_length=self.max_length,
            use_cache=False,
        )
        collator = DynamicPaddingCollator(pad_token_id=dataset.tokenizer.pad_token_id)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collator)

        all_logits = []
        all_labels = dataset.labels

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.extend(outputs.logits.cpu().tolist())

        return np.array(all_logits), np.array(all_labels)

    def calibrate(self) -> dict[str, Any]:
        """Calibrates the model using temperature scaling on val logits."""
        logger.info("Extracting validation and test logits...")
        val_logits, val_labels = self.extract_logits(self.val_df)
        test_logits, test_labels = self.extract_logits(self.test_df)

        logger.info("Optimising temperature scaling on validation set...")
        val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
        val_labels_t = torch.tensor(val_labels, dtype=torch.long)

        scaler = TemperatureScaler()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=100)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(scaler(val_logits_t), val_labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        opt_temp = float(scaler.temperature.item())
        logger.info("Optimised temperature: %.4f", opt_temp)

        # Apply scaling to test logits
        test_logits_t = torch.tensor(test_logits, dtype=torch.float32)
        calibrated_test_logits = scaler(test_logits_t).detach().numpy()

        # Compute probabilities before and after calibration
        probs_before = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
        probs_after = torch.softmax(torch.tensor(calibrated_test_logits), dim=-1).numpy()

        # ECE calculations
        ece_before = calculate_ece(probs_before, test_labels)
        ece_after = calculate_ece(probs_after, test_labels)

        # Brier Score calculations
        brier_before = calculate_brier_score(probs_before, test_labels)
        brier_after = calculate_brier_score(probs_after, test_labels)

        logger.info("Before Calibration: ECE = %.4f | Brier = %.4f", ece_before, brier_before)
        logger.info("After Calibration:  ECE = %.4f | Brier = %.4f", ece_after, brier_after)

        metrics = {
            "optimized_temperature": opt_temp,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "brier_before": brier_before,
            "brier_after": brier_after,
        }
        save_metrics(metrics, self.metrics_dir / "calibration_summary.json")

        # 4. Generate Diagnostic Diagrams
        self._plot_reliability_diagram(probs_before, probs_after, test_labels)
        self._plot_confidence_histogram(probs_before, probs_after)
        self._plot_threshold_study(probs_before, probs_after, test_labels)

        return metrics

    def _plot_reliability_diagram(
        self, probs_before: np.ndarray, probs_after: np.ndarray, labels: np.ndarray
    ) -> None:
        """Plots confidence vs accuracy reliability diagram comparing before and after."""
        num_bins = 10
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])

        def get_bin_accuracies_and_confs(probs):
            confidences = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)
            accuracies = predictions == labels

            bin_accs = []
            bin_confs = []
            for i in range(num_bins):
                in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
                if np.sum(in_bin) > 0:
                    bin_accs.append(np.mean(accuracies[in_bin]))
                    bin_confs.append(np.mean(confidences[in_bin]))
                else:
                    bin_accs.append(0.0)
                    bin_confs.append(bin_centers[i])
            return np.array(bin_accs), np.array(bin_confs)

        acc_before, conf_before = get_bin_accuracies_and_confs(probs_before)
        acc_after, conf_after = get_bin_accuracies_and_confs(probs_after)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        ax.plot(conf_before, acc_before, "r-s", label="Before Calibration (Uncalibrated)")
        ax.plot(conf_after, acc_after, "g-o", label="After Calibration (Temperature Scaled)")

        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Fraction of Correct Predictions (Accuracy)")
        ax.set_title("Reliability Diagram Comparison")
        ax.legend(loc="upper left")
        ax.grid(True)
        plt.tight_layout()
        save_figure(fig, self.plots_dir / "reliability_diagram.png")
        plt.close()

    def _plot_confidence_histogram(self, probs_before: np.ndarray, probs_after: np.ndarray) -> None:
        """Plots frequency density of prediction confidences before and after."""
        conf_before = np.max(probs_before, axis=1)
        conf_after = np.max(probs_after, axis=1)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(
            conf_before,
            bins=20,
            alpha=0.5,
            color="red",
            label="Before Calibration",
            density=True,
        )
        ax.hist(
            conf_after,
            bins=20,
            alpha=0.5,
            color="green",
            label="After Calibration",
            density=True,
        )

        ax.set_xlabel("Maximum Prediction Confidence probability")
        ax.set_ylabel("Density")
        ax.set_title("Prediction Confidences Distribution Density")
        ax.legend(loc="upper left")
        ax.grid(True)
        plt.tight_layout()
        save_figure(fig, self.plots_dir / "confidence_histogram.png")
        plt.close()

    def _plot_threshold_study(
        self, probs_before: np.ndarray, probs_after: np.ndarray, labels: np.ndarray
    ) -> None:
        """Plots prediction accuracy vs sample coverage curves at varying thresholds."""
        thresholds = np.linspace(0, 0.95, 20)

        def get_curves(probs):
            confidences = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)
            accuracies = predictions == labels

            acc_curve = []
            cov_curve = []
            for t in thresholds:
                passed = confidences >= t
                coverage = np.mean(passed)
                accuracy = np.mean(accuracies[passed]) if np.sum(passed) > 0 else 1.0

                acc_curve.append(accuracy)
                cov_curve.append(coverage)

            return np.array(acc_curve), np.array(cov_curve)

        acc_before, cov_before = get_curves(probs_before)
        acc_after, cov_after = get_curves(probs_after)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(thresholds, acc_before * 100, "r--", label="Accuracy (Before)")
        ax.plot(thresholds, cov_before * 100, "r-", label="Coverage (Before)")
        ax.plot(thresholds, acc_after * 100, "g--", label="Accuracy (After Cal)")
        ax.plot(thresholds, cov_after * 100, "g-", label="Coverage (After Cal)")

        ax.set_xlabel("Confidence Routing Threshold")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Threshold Decision Study (Calibrated vs Uncalibrated)")
        ax.legend(loc="lower left")
        ax.grid(True)
        plt.tight_layout()
        save_figure(fig, self.plots_dir / "threshold_study.png")
        plt.close()


def main() -> None:
    """CLI entrypoint for running calibration optimization."""
    parser = argparse.ArgumentParser(
        description="Run temperature scaling calibration on sequence classifier."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(OUTPUT_DIR / "models" / "best_model"),
        help="Path to fine-tuned transformer model.",
    )
    parser.add_argument(
        "--smoke-run",
        action="store_true",
        help="Calibrate on a tiny subset of inputs.",
    )
    args = parser.parse_args()

    try:
        runner = CalibrationRunner(args.model_dir, smoke_run=args.smoke_run)
        runner.calibrate()
    except Exception as e:
        logger.exception("Calibration runner execution failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
