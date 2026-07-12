"""
decision_engine.py
==================
Implements a 3-tier customer support ticket routing and decision engine:
1. High Confidence: Automated routing to classifier intent.
2. Mid Confidence: Retrieval of similar tickets + LLM draft reply generation.
3. Low Confidence: Immediate escalation to human support review.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from src.models.transformer.retrieval import SemanticRetriever
from src.utils.config import load_config
from src.utils.constants import OUTPUT_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DecisionEngine:
    """Orchestrates 3-tier routing, semantic search context, and LLM text generation."""

    def __init__(
        self,
        model_dir: Path | str,
        retriever_index_dir: Path | str,
        config: dict[str, Any],
        device: str | None = None,
    ) -> None:
        """Initialises the Decision Engine and loads all models into memory.

        Args:
            model_dir: Path to the fine-tuned classifier model directory.
            retriever_index_dir: Path to the FAISS retriever index directory.
            config: Configuration dictionary.
            device: Device ('cuda', 'cpu', or None to auto-detect).
        """
        self.config = config
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_str)

        # 1. Routing thresholds
        de_config = config.get("decision_engine", {})
        self.high_threshold = float(de_config.get("high_confidence_threshold", 0.90))
        self.low_threshold = float(de_config.get("low_confidence_threshold", 0.60))

        # 2. Load Classifier model & tokenizer
        self.model_dir = Path(model_dir)
        logger.info("Loading intent classifier from: %s", self.model_dir)
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir)
        )
        self.classifier_model.to(self.device)
        self.classifier_model.eval()

        # Load label mapping
        self.id_to_label: dict[int, str] = {}
        encoder_path = OUTPUT_DIR / "models" / "label_encoder.json"
        if encoder_path.exists():
            try:
                with open(encoder_path, encoding="utf-8") as f:
                    mapping = json.load(f)
                self.id_to_label = {int(k): v for k, v in mapping.get("id_to_label", {}).items()}
            except Exception as e:
                logger.warning("Could not load label encoder: %s", e)

        if not self.id_to_label:
            if self.classifier_model.config.id2label:
                self.id_to_label = {
                    int(k): v for k, v in self.classifier_model.config.id2label.items()
                }
            else:
                self.id_to_label = {
                    i: f"LABEL_{i}" for i in range(self.classifier_model.config.num_labels)
                }

        # Load calibrated temperature if available
        self.temperature = 1.0
        calib_path = OUTPUT_DIR / "metrics" / "calibration_summary.json"
        if calib_path.exists():
            try:
                with open(calib_path, encoding="utf-8") as f:
                    summary = json.load(f)
                self.temperature = float(summary.get("optimized_temperature", 1.0))
                logger.info("Loaded calibrated temperature scaling: T = %.4f", self.temperature)
            except Exception as e:
                logger.warning("Could not load calibration summary: %s", e)

        # 3. Load Retriever index
        self.retriever_index_dir = Path(retriever_index_dir)
        logger.info("Loading semantic retriever from: %s", self.retriever_index_dir)
        self.retriever = SemanticRetriever(device=self.device_str)
        self.retriever.load_index(self.retriever_index_dir)

        # 4. Load LLM Backend
        llm_config = config.get("llm", {})
        self.llm_enabled = llm_config.get("enabled", True)
        self.llm_model_id = llm_config.get("model_id", "microsoft/Phi-3-mini-4k-instruct")
        self.max_new_tokens = int(llm_config.get("max_new_tokens", 128))
        self.llm_temp = float(llm_config.get("temperature", 0.2))

        self.llm_pipeline = None
        if self.llm_enabled:
            logger.info(
                "Loading LLM backend model ONCE into cache: %s (low_cpu_mem_usage=True)",
                self.llm_model_id,
            )
            try:
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    self.llm_model_id, trust_remote_code=True
                )
                # Pad token configuration
                if self.llm_tokenizer.pad_token is None:
                    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                self.llm_model.to(self.device)
                self.llm_model.eval()

                self.llm_pipeline = pipeline(
                    "text-generation",
                    model=self.llm_model,
                    tokenizer=self.llm_tokenizer,
                    device=self.device,
                )
            except Exception as e:
                logger.error("Failed to initialise LLM backend %s: %s", self.llm_model_id, e)
                # Fallback to disabled LLM state on error
                self.llm_enabled = False

    def predict_intent(self, text: str) -> tuple[str, float]:
        """Runs the intent classifier on the input text to output label and confidence.

        Args:
            text: Input ticket text.

        Returns:
            Tuple of (predicted_label, confidence).
        """
        inputs = self.classifier_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.classifier_model(**inputs)

        # Apply temperature scaling to logits
        calibrated_logits = outputs.logits / self.temperature
        probs = torch.softmax(calibrated_logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_label = self.id_to_label[pred_idx]

        return pred_label, confidence

    def generate_draft_reply(
        self,
        ticket_text: str,
        predicted_intent: str,
        confidence: float,
        retrieved_docs: list[dict[str, Any]],
    ) -> str:
        """Generates a contextual reply draft using the cached LLM pipeline.

        Args:
            ticket_text: The user ticket text.
            predicted_intent: Predicted ticket class.
            confidence: Predicted classification confidence score.
            retrieved_docs: List of retrieved similar support tickets.

        Returns:
            Generated response string.
        """
        if not self.llm_enabled or self.llm_pipeline is None:
            return "LLM generator is currently disabled or unavailable."

        # Build prompt using retrieved cases and intent metadata
        cases_text = ""
        for i, doc in enumerate(retrieved_docs, 1):
            cases_text += f"{i}. {doc['text']}\n"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful customer support assistant. Draft a concise reply "
                    "to the Customer Ticket using the provided similar historical cases."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Customer Ticket:\n{ticket_text}\n\n"
                    f"Predicted Intent: {predicted_intent}\n"
                    f"Confidence: {confidence:.4f}\n\n"
                    f"Similar Historical Cases:\n{cases_text}\n"
                    f"Generate a concise draft reply."
                ),
            },
        ]

        try:
            # Use tokenizer chat template if available
            prompt = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback plain prompt structure
            prompt = (
                f"System: You are a helpful customer support assistant. Draft a concise reply "
                f"to the Customer Ticket using similar historical cases.\n\n"
                f"Customer Ticket:\n{ticket_text}\n\n"
                f"Predicted Intent: {predicted_intent}\n"
                f"Confidence: {confidence:.4f}\n\n"
                f"Similar Historical Cases:\n{cases_text}\n"
                f"Assistant:"
            )

        # Generate reply
        outputs = self.llm_pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.llm_temp,
            do_sample=self.llm_temp > 0.0,
            pad_token_id=self.llm_tokenizer.pad_token_id,
        )
        full_text = outputs[0]["generated_text"]

        # Clean generated text to extract assistant portion only
        if "Assistant:" in full_text:
            reply = full_text.split("Assistant:")[-1].strip()
        elif "<|assistant|>" in full_text:
            reply = full_text.split("<|assistant|>")[-1].strip()
        else:
            reply = full_text[len(prompt) :].strip()

        return reply

    def route_ticket(self, text: str) -> dict[str, Any]:
        """Routes support ticket using the 3-tier decision engine logic.

        Args:
            text: Customer ticket text.

        Returns:
            Dictionary representing structured routing response.
        """
        # Step 1: Predict intent and confidence
        predicted_intent, confidence = self.predict_intent(text)

        # Tier 1: High Confidence
        if confidence >= self.high_threshold:
            return {
                "intent": predicted_intent,
                "confidence": confidence,
                "route": "classifier",
                "retrieved_docs": [],
                "llm_used": False,
                "reply": f"Automated routing to category: {predicted_intent}",
            }

        # Tier 2: Mid Confidence (LLM Fallback + Retrieval)
        if confidence >= self.low_threshold:
            retrieved = self.retriever.retrieve(text, top_k=3)
            reply_draft = self.generate_draft_reply(
                ticket_text=text,
                predicted_intent=predicted_intent,
                confidence=confidence,
                retrieved_docs=retrieved,
            )
            return {
                "intent": predicted_intent,
                "confidence": confidence,
                "route": "fallback",
                "retrieved_docs": retrieved,
                "llm_used": True,
                "reply": reply_draft,
            }

        # Tier 3: Low Confidence (Escalation to Human)
        return {
            "intent": "unknown",
            "confidence": confidence,
            "route": "human_escalation",
            "retrieved_docs": [],
            "llm_used": False,
            "reply": f"Escalated to human support review due to low confidence ({confidence:.4f}).",
        }


def main() -> None:
    """CLI entrypoint to run routing decision engine over a customer support ticket."""
    parser = argparse.ArgumentParser(description="Run SupportAI Decision Engine routing.")
    parser.add_argument("--text", type=str, required=True, help="Customer ticket text.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(OUTPUT_DIR / "models" / "best_model"),
        help="Classifier model directory.",
    )
    parser.add_argument(
        "--retriever_index_dir",
        type=str,
        default=str(OUTPUT_DIR / "retrieval_index"),
        help="FAISS index directory.",
    )
    parser.add_argument(
        "--high_threshold", type=float, default=None, help="High confidence threshold."
    )
    parser.add_argument(
        "--low_threshold", type=float, default=None, help="Low confidence threshold."
    )
    parser.add_argument(
        "--config_overlay", type=str, default=None, help="YAML config overlay file."
    )
    parser.add_argument("--llm_model_id", type=str, default=None, help="LLM model ID override.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_overlay)

    # Overwrite config values with command line overrides if provided
    if "decision_engine" not in config:
        config["decision_engine"] = {}
    if args.high_threshold is not None:
        config["decision_engine"]["high_confidence_threshold"] = args.high_threshold
    if args.low_threshold is not None:
        config["decision_engine"]["low_confidence_threshold"] = args.low_threshold

    if "llm" not in config:
        config["llm"] = {}
    if args.llm_model_id is not None:
        config["llm"]["model_id"] = args.llm_model_id

    # Initialize DecisionEngine
    engine = DecisionEngine(
        model_dir=args.model_dir,
        retriever_index_dir=args.retriever_index_dir,
        config=config,
    )

    # Route ticket
    result = engine.route_ticket(args.text)

    # Print JSON output
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
