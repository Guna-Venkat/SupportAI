"""
explainability.py
=================
Provides modular explainability utilities using LIME
(Local Interpretable Model-agnostic Explanations)
for the fine-tuned customer support sequence classifier.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.constants import OUTPUT_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)
console = Console()


class TicketExplainer:
    """Manages explainability for support tickets using LIME."""

    def __init__(self, model_dir: Path | str, device: str | None = None) -> None:
        """Initialises the explainer with a saved model, tokenizer, and labels.

        Args:
            model_dir: Path to the HF model directory.
            device: Computing device ('cuda', 'cpu', or None to auto-detect).
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        logger.info("Loading model and tokenizer for explainability from %s", self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

        # Load label decoder
        self.encoder_path = OUTPUT_DIR / "models" / "label_encoder.json"
        self.id_to_label: dict[int, str] = {}
        if self.encoder_path.exists():
            try:
                with open(self.encoder_path, encoding="utf-8") as f:
                    mapping = json.load(f)
                self.id_to_label = {int(k): v for k, v in mapping.get("id_to_label", {}).items()}
            except Exception as e:
                logger.warning("Could not load label encoder from %s: %s", self.encoder_path, e)

        # Fallback to model config mapping
        if not self.id_to_label:
            if self.model.config.id2label:
                self.id_to_label = {int(k): v for k, v in self.model.config.id2label.items()}
            else:
                self.id_to_label = {i: f"LABEL_{i}" for i in range(self.model.config.num_labels)}

        # Sort classes by index to align with predict_proba outputs
        self.class_names = [self.id_to_label[i] for i in sorted(self.id_to_label.keys())]

        # Initialise LIME explainer
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        """Generates probabilities for a list of perturbed texts.

        Args:
            texts: List of perturbed input strings.

        Returns:
            A numpy array of shape (num_texts, num_classes) with probabilities.
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        return probs

    def explain_ticket(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 100,
    ) -> dict[str, Any]:
        """Explains prediction on a given support ticket text.

        Args:
            text: Ticket text string to explain.
            num_features: Maximum number of features (words) to include.
            num_samples: Number of perturbation samples to generate.

        Returns:
            A dictionary containing predicted class details, word attributions, and HTML report.
        """
        # Get raw probabilities
        probs = self._predict_proba([text])[0]
        pred_class_idx = int(np.argmax(probs))
        pred_label = self.class_names[pred_class_idx]
        pred_prob = float(probs[pred_class_idx])

        # Explain prediction
        explanation = self.explainer.explain_instance(
            text,
            self._predict_proba,
            labels=(pred_class_idx,),
            num_features=num_features,
            num_samples=num_samples,
        )

        # Get attributions for the predicted class
        attributions = explanation.as_list(label=pred_class_idx)
        explanation_html = explanation.as_html()

        return {
            "predicted_class": pred_label,
            "predicted_probability": pred_prob,
            "attributions": attributions,
            "explanation_html": explanation_html,
        }

    def visualize_explanation(
        self,
        explanation_result: dict[str, Any],
        top_k: int = 10,
    ) -> None:
        """Prints a beautiful console visualizer of the prediction explanation.

        Args:
            explanation_result: Dictionary returned by explain_ticket().
            top_k: Number of attributions to list.
        """
        pred_class = explanation_result["predicted_class"]
        pred_prob = explanation_result["predicted_probability"]
        attributions = explanation_result["attributions"][:top_k]

        console.print()
        console.print("[bold cyan]Prediction Explanation Summary[/bold cyan]")
        console.print(f"Predicted Class: [bold yellow]{pred_class}[/bold yellow]")
        console.print(f"Confidence Score: [bold green]{pred_prob:.2%}[/bold green]")
        console.print()

        table = Table(
            title=f"Word Attributions for Class '{pred_class}'",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Rank", justify="center")
        table.add_column("Word / Feature", justify="left")
        table.add_column("Attribution Score", justify="right")
        table.add_column("Direction", justify="center")

        for idx, (word, score) in enumerate(attributions, 1):
            direction = (
                "[bold green]Positive[/bold green]"
                if score > 0
                else "[bold red]Negative[/bold red]"
            )
            score_color = "green" if score > 0 else "red"
            table.add_row(
                str(idx),
                word,
                f"[{score_color}]{score:+.4f}[/{score_color}]",
                direction,
            )

        console.print(table)
        console.print()


def main() -> None:
    """CLI entrypoint for explaining sequence classifier predictions."""
    parser = argparse.ArgumentParser(
        description="Explain customer ticket classification predictions."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(OUTPUT_DIR / "models" / "best_model"),
        help="Path to saved transformer model directory.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Ticket text to explain.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of perturbation samples for LIME.",
    )
    parser.add_argument(
        "--output_html",
        type=str,
        default=None,
        help="Optional file path to save LIME HTML report.",
    )
    args = parser.parse_args()

    # Load explainer
    model_path = Path(args.model_dir)
    if not model_path.exists():
        console.print(f"[bold red]Error: Model path {model_path} does not exist.[/bold red]")
        return

    explainer = TicketExplainer(model_dir=model_path)
    result = explainer.explain_ticket(args.text, num_samples=args.num_samples)

    explainer.visualize_explanation(result)

    if args.output_html:
        output_path = Path(args.output_html)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result["explanation_html"], encoding="utf-8")
        console.print(f"[bold green]Saved HTML report to {output_path}[/bold green]\n")


if __name__ == "__main__":
    main()
