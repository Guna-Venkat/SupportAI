"""
train.py
========
Fine-tuning pipeline for DistilBERT on the customer support dataset.
Supports mixed precision (AMP), gradient accumulation, early stopping,
checkpoint saving/resuming, and MLflow logging.
"""

import argparse
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from src.data.dataset import load_and_preprocess_dataset
from src.evaluation.metrics import calculate_metrics
from src.models.transformer.collator import DynamicPaddingCollator
from src.models.transformer.dataset import TransformerTicketDataset
from src.utils.artifacts import save_csv, save_json, save_metrics
from src.utils.config import load_config
from src.utils.constants import OUTPUT_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def train_model(
    config_overlay: Path | str | None = None,
    smoke_run: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Fine-tunes a DistilBERT model based on the centralized configuration.

    Args:
        config_overlay: Optional path to config overlay file.
        smoke_run: If True, trains for 1 epoch on a tiny subset of data.
        resume: If True, attempts to resume training from an existing checkpoint.

    Returns:
        Evaluation metrics dictionary on the test split.
    """
    config = load_config(config_overlay)

    # 1. Setup paths
    models_dir = OUTPUT_DIR / "models"
    checkpoints_dir = OUTPUT_DIR / "checkpoints"
    metrics_dir = OUTPUT_DIR / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / "transformer_checkpoint.pt"
    best_model_dir = models_dir / "best_model"

    # 2. Extract configuration hyper-parameters
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    model_name = config.get("model", {}).get("name", "distilbert-base-uncased")
    # Support fallback name mapping
    if model_name == "tfidf_classifier":
        model_name = "distilbert-base-uncased"

    # Support both nested 'train' dictionary overrides (from train.yaml) and top-level defaults
    train_cfg = config.get("train", {})
    epochs = 1 if smoke_run else train_cfg.get("epochs", config.get("epochs", 3))
    batch_size = 4 if smoke_run else train_cfg.get("batch_size", config.get("batch_size", 16))
    lr = float(train_cfg.get("learning_rate", config.get("learning_rate", 2e-5)))
    weight_decay = float(train_cfg.get("weight_decay", config.get("weight_decay", 0.01)))
    max_length = config.get("max_length", 128)
    patience = train_cfg.get(
        "early_stopping_patience", train_cfg.get("patience", config.get("patience", 3))
    )
    gradient_accumulation_steps = train_cfg.get(
        "gradient_accumulation_steps", config.get("gradient_accumulation_steps", 1)
    )

    # 3. Load dataset splits
    splits = load_and_preprocess_dataset(config_overlay)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # Determine number of classes from full splits to avoid out-of-bounds error on sliced data
    num_labels = int(
        max(train_df["label"].max(), val_df["label"].max(), test_df["label"].max()) + 1
    )

    if smoke_run:
        train_df = train_df.head(16)
        val_df = val_df.head(8)
        test_df = test_df.head(8)

    logger.info("Intents class count: %d", num_labels)

    # 4. Initialize tokenized PyTorch Datasets
    logger.info("Initializing train and val transformer datasets...")
    train_dataset = TransformerTicketDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        model_name=model_name,
        max_length=max_length,
        cache_path=None if smoke_run else (OUTPUT_DIR / "transformer_cache" / "train.pt"),
    )
    val_dataset = TransformerTicketDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        model_name=model_name,
        max_length=max_length,
        cache_path=None if smoke_run else (OUTPUT_DIR / "transformer_cache" / "val.pt"),
    )
    test_dataset = TransformerTicketDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        model_name=model_name,
        max_length=max_length,
        cache_path=None if smoke_run else (OUTPUT_DIR / "transformer_cache" / "test.pt"),
    )

    # 5. Initialize PyTorch DataLoaders with dynamic padding collator
    collator = DynamicPaddingCollator(pad_token_id=train_dataset.tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    # 6. Initialize Model, Optimizer, Scheduler, and AMP Scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Calculate scheduler warmup and total training steps
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Mixed precision setup
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Print active hyperparameters at startup
    config_block = (
        "\n==========================\n"
        "Training Configuration\n"
        "==========================\n"
        f"Model: {model_name}\n"
        f"Train samples: {len(train_df)}\n"
        f"Validation samples: {len(val_df)}\n"
        f"Epochs: {epochs}\n"
        f"Batch Size: {batch_size}\n"
        f"Learning Rate: {lr:.0e}\n"
        f"Weight Decay: {weight_decay}\n"
        f"Warmup Ratio: 0.10\n"
        f"Scheduler: Linear\n"
        f"Max Length: {max_length}\n"
        f"Device: {device.type}\n"
        f"Mixed Precision: {use_amp}\n"
        f"Gradient Accumulation: {gradient_accumulation_steps}\n"
        f"Early Stopping: {patience}\n"
        f"Seed: {seed}\n"
        "=========================="
    )
    logger.info(config_block)

    # 7. Resume Checkpoint loading
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    if resume and checkpoint_path.exists():
        logger.info("Found existing checkpoint. Loading from: %s", checkpoint_path)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if scaler is not None and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            patience_counter = checkpoint["patience_counter"]
            history = checkpoint.get("history", [])
            logger.info("Resumed successfully from epoch %d.", start_epoch)
        except Exception as e:
            logger.warning("Could not load checkpoint: %s. Starting from scratch.", e)

    # Start MLflow run
    mlflow.set_experiment("DistilBERT_Fine_Tuning")
    with mlflow.start_run(run_name="distilbert_run", nested=True):
        mlflow.log_params(
            {
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "max_length": max_length,
                "patience": patience,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "device": str(device),
                "use_amp": use_amp,
            }
        )

        # 8. Training & Validation Epoch Loop
        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            logger.info("Epoch %d/%d - Training...", epoch + 1, epochs)
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass with AMP autocast
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    # Scale loss according to accumulation steps
                    loss = loss / gradient_accumulation_steps

                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                train_loss += loss.item() * gradient_accumulation_steps

                # Optimizer step execution
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()

            avg_train_loss = train_loss / len(train_loader)
            logger.info("Epoch %d/%d - Avg Train Loss: %.4f", epoch + 1, epochs, avg_train_loss)

            # Validation stage
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            logger.info("Epoch %d/%d - Evaluating...", epoch + 1, epochs)
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = outputs.loss

                    val_loss += loss.item()
                    preds = torch.argmax(outputs.logits, dim=-1)
                    val_preds.extend(preds.cpu().tolist())
                    val_targets.extend(labels.cpu().tolist())

            avg_val_loss = val_loss / len(val_loader)
            val_metrics = calculate_metrics(val_targets, val_preds, average="weighted")
            avg_val_acc = val_metrics["accuracy"]

            logger.info(
                "Epoch %d/%d - Val Loss: %.4f | Val Acc: %.2f%%",
                epoch + 1,
                epochs,
                avg_val_loss,
                avg_val_acc * 100,
            )

            # Record epoch statistics
            epoch_history = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": avg_val_acc,
            }
            history.append(epoch_history)

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": avg_val_acc,
                },
                step=epoch + 1,
            )

            # Export learning curves to csv and json
            save_csv(pd.DataFrame(history), metrics_dir / "training_history.csv")
            save_json(history, metrics_dir / "training_history.json")

            # 9. Early Stopping & Checkpoint saving logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                logger.info("New best model validation loss! Saving best weights...")
                model.save_pretrained(best_model_dir)
                train_dataset.tokenizer.save_pretrained(best_model_dir)
            else:
                patience_counter += 1
                logger.info(
                    "Validation loss did not improve. Patience: %d/%d", patience_counter, patience
                )

            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                    "history": history,
                },
                checkpoint_path,
            )

            # Check if early stopping is triggered
            if patience_counter >= patience:
                logger.info("Early stopping triggered. Terminating train loop.")
                break

        # Remove checkpoint file if finished training epochs normally without trigger early stopping
        if checkpoint_path.exists() and patience_counter < patience and not smoke_run:
            try:
                checkpoint_path.unlink()
                logger.info("Training complete. Cleaned up checkpoint file.")
            except Exception as e:
                logger.warning("Could not delete checkpoint file: %s", e)

        # 10. Load best model and evaluate on the test split
        logger.info("Loading best model weights for final evaluation on test split...")
        best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
        best_model.to(device)
        best_model.eval()

        test_preds = []
        test_targets = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = best_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                preds = torch.argmax(outputs.logits, dim=-1)
                test_preds.extend(preds.cpu().tolist())

            if "labels" in batch:
                # Retrieve all targets from test dataset labels
                test_targets = test_dataset.labels

        test_metrics = calculate_metrics(test_targets, test_preds, average="weighted")
        save_metrics(test_metrics, metrics_dir / "metrics.json")
        logger.info("Final test set accuracy: %.2f%%", test_metrics["accuracy"] * 100)

        mlflow.log_metrics(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1_weighted"],
            }
        )

    return test_metrics


def main() -> None:
    """CLI entrypoint for transformer training execution."""
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on Banking77.")
    parser.add_argument("--config", type=str, default=None, help="Path to config overlay.")
    parser.add_argument(
        "--smoke-run",
        action="store_true",
        help="Train for 1 epoch on a tiny subset of inputs.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume training from existing checkpoint.",
    )
    args = parser.parse_args()

    train_model(
        config_overlay=args.config,
        smoke_run=args.smoke_run,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
