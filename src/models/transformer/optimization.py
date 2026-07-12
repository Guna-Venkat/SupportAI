"""
optimization.py
===============
Handles ONNX export, dynamic INT8 quantization (PyTorch & ONNX), and comprehensive
performance profiling (Latency, Throughput, RAM, Disk Size, Cold Start).
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import psutil
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.dataset import load_and_preprocess_dataset
from src.models.transformer.collator import DynamicPaddingCollator
from src.models.transformer.dataset import TransformerTicketDataset
from src.utils.artifacts import save_metrics
from src.utils.config import load_config
from src.utils.constants import OUTPUT_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ONNXModelWrapper(torch.nn.Module):
    """Wrapper class to return only logits tensor from sequence classifier."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


def get_file_size_mb(path: Path) -> float:
    """Returns file size in megabytes."""
    if not path.exists():
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def get_process_ram_mb() -> float:
    """Returns the resident set size (RSS) memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_pytorch_inference(model: Any, batch: dict[str, torch.Tensor]) -> None:
    """Runs a forward pass on PyTorch model."""
    with torch.no_grad():
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])


def run_onnx_inference(session: Any, batch: dict[str, torch.Tensor]) -> None:
    """Runs forward pass on ONNX Runtime session."""
    inputs = {
        "input_ids": batch["input_ids"].numpy().astype(np.int64),
        "attention_mask": batch["attention_mask"].numpy().astype(np.int64),
    }
    session.run(None, inputs)


class ModelOptimizer:
    """Handles exporting, quantising, and profiling sequence classification models."""

    def __init__(
        self,
        model_dir: Path | str,
        config_overlay: Path | str | None = None,
        smoke_run: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config = load_config(config_overlay)
        self.smoke_run = smoke_run

        self.models_dir = OUTPUT_DIR / "models"
        self.metrics_dir = OUTPUT_DIR / "metrics"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.onnx_path = self.models_dir / "model.onnx"
        self.onnx_quant_path = self.models_dir / "model_quant.onnx"
        self.pytorch_quant_path = self.models_dir / "pytorch_quant.pt"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

    def export_onnx(self) -> Path:
        """Exports fine-tuned sequence classifier to ONNX format."""
        logger.info("Exporting model to ONNX format...")
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        model.eval()

        wrapper = ONNXModelWrapper(model)
        dummy_input_ids = torch.ones(1, 16, dtype=torch.long)
        dummy_attention_mask = torch.ones(1, 16, dtype=torch.long)
        dummy_input = (dummy_input_ids, dummy_attention_mask)

        torch.onnx.export(
            wrapper,
            dummy_input,
            str(self.onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            opset_version=18,
        )

        # Load exported ONNX model and clear conflicting value_info to bypass ORT quantization bugs
        onnx_model = onnx.load(str(self.onnx_path))
        del onnx_model.graph.value_info[:]
        onnx.save(onnx_model, str(self.onnx_path))

        logger.info("ONNX model saved to: %s", self.onnx_path)
        return self.onnx_path

    def quantize_pytorch(self) -> Path:
        """Applies dynamic quantization to PyTorch FP32 model and saves weight dict."""
        logger.info("Applying dynamic quantization to PyTorch model...")
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        model.eval()

        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        torch.save(quantized_model.state_dict(), self.pytorch_quant_path)
        logger.info("Quantized PyTorch model saved to: %s", self.pytorch_quant_path)
        return self.pytorch_quant_path

    def quantize_onnx(self) -> Path:
        """Applies dynamic weight quantization to exported ONNX model."""
        logger.info("Applying dynamic quantization to ONNX model...")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(
                model_input=str(self.onnx_path),
                model_output=str(self.onnx_quant_path),
                weight_type=QuantType.QUInt8,
            )
            logger.info("Quantized ONNX model saved to: %s", self.onnx_quant_path)
        except Exception as e:
            logger.warning("Could not quantize ONNX model: %s", e)
        return self.onnx_quant_path

    def benchmark_models(self) -> dict[str, Any]:
        """Runs performance diagnostics comparing PyTorch, ONNX, and INT8 models."""
        logger.info("Starting performance benchmarking suite...")

        splits = load_and_preprocess_dataset(None)
        test_df = splits["test"]
        if self.smoke_run:
            test_df = test_df.head(10)

        # Prepare test dataset & loader
        dataset = TransformerTicketDataset(
            texts=test_df["text"].tolist(),
            labels=test_df["label"].tolist(),
            model_name=str(self.model_dir),
            max_length=self.config.get("max_length", 128),
            use_cache=False,
        )
        collator = DynamicPaddingCollator(pad_token_id=dataset.tokenizer.pad_token_id)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
        batches = list(loader)

        results = {}

        # 1. Benchmarking PyTorch (FP32)
        logger.info("Benchmarking PyTorch (FP32)...")
        t0 = time.perf_counter()
        model_pt = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        model_pt.eval()
        cold_start_pt = time.perf_counter() - t0
        disk_pt = get_file_size_mb(self.model_dir / "model.safetensors")

        ram_before = get_process_ram_mb()
        # Warmup
        run_pytorch_inference(model_pt, batches[0])
        ram_after = get_process_ram_mb()
        ram_pt = max(0.0, ram_after - ram_before)

        latencies_pt = []
        for batch in batches:
            t_start = time.perf_counter()
            run_pytorch_inference(model_pt, batch)
            latencies_pt.append(time.perf_counter() - t_start)

        avg_latency_pt = float(np.mean(latencies_pt) * 1000)  # ms
        throughput_pt = 1.0 / np.mean(latencies_pt) if latencies_pt else 0.0

        results["PyTorch_FP32"] = {
            "Latency (ms)": avg_latency_pt,
            "Throughput (QPS)": throughput_pt,
            "Disk Size (MB)": disk_pt,
            "RAM Usage (MB)": ram_pt,
            "Cold Start (s)": cold_start_pt,
        }

        # 2. Benchmarking PyTorch Quantized (INT8)
        logger.info("Benchmarking PyTorch Quantized (INT8)...")
        t0 = time.perf_counter()
        model_pt_quant = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        model_pt_quant = torch.quantization.quantize_dynamic(
            model_pt_quant, {torch.nn.Linear}, dtype=torch.qint8
        )
        if self.pytorch_quant_path.exists():
            model_pt_quant.load_state_dict(torch.load(self.pytorch_quant_path, weights_only=True))
        model_pt_quant.eval()
        cold_start_pt_quant = time.perf_counter() - t0
        disk_pt_quant = get_file_size_mb(self.pytorch_quant_path)

        ram_before = get_process_ram_mb()
        run_pytorch_inference(model_pt_quant, batches[0])
        ram_after = get_process_ram_mb()
        ram_pt_quant = max(0.0, ram_after - ram_before)

        latencies_pt_quant = []
        for batch in batches:
            t_start = time.perf_counter()
            run_pytorch_inference(model_pt_quant, batch)
            latencies_pt_quant.append(time.perf_counter() - t_start)

        avg_latency_pt_quant = float(np.mean(latencies_pt_quant) * 1000)
        throughput_pt_quant = 1.0 / np.mean(latencies_pt_quant) if latencies_pt_quant else 0.0

        results["PyTorch_INT8"] = {
            "Latency (ms)": avg_latency_pt_quant,
            "Throughput (QPS)": throughput_pt_quant,
            "Disk Size (MB)": disk_pt_quant,
            "RAM Usage (MB)": ram_pt_quant,
            "Cold Start (s)": cold_start_pt_quant,
        }

        # 3. Benchmarking ONNX (FP32)
        logger.info("Benchmarking ONNX (FP32)...")
        import onnxruntime as ort

        t0 = time.perf_counter()
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        session_onnx = ort.InferenceSession(str(self.onnx_path), sess_options)
        cold_start_onnx = time.perf_counter() - t0
        disk_onnx = get_file_size_mb(self.onnx_path)

        ram_before = get_process_ram_mb()
        run_onnx_inference(session_onnx, batches[0])
        ram_after = get_process_ram_mb()
        ram_onnx = max(0.0, ram_after - ram_before)

        latencies_onnx = []
        for batch in batches:
            t_start = time.perf_counter()
            run_onnx_inference(session_onnx, batch)
            latencies_onnx.append(time.perf_counter() - t_start)

        avg_latency_onnx = float(np.mean(latencies_onnx) * 1000)
        throughput_onnx = 1.0 / np.mean(latencies_onnx) if latencies_onnx else 0.0

        results["ONNX_FP32"] = {
            "Latency (ms)": avg_latency_onnx,
            "Throughput (QPS)": throughput_onnx,
            "Disk Size (MB)": disk_onnx,
            "RAM Usage (MB)": ram_onnx,
            "Cold Start (s)": cold_start_onnx,
        }

        # 4. Benchmarking ONNX Quantized (INT8)
        if self.onnx_quant_path.exists():
            logger.info("Benchmarking ONNX Quantized (INT8)...")
            t0 = time.perf_counter()
            session_onnx_quant = ort.InferenceSession(str(self.onnx_quant_path), sess_options)
            cold_start_onnx_quant = time.perf_counter() - t0
            disk_onnx_quant = get_file_size_mb(self.onnx_quant_path)

            ram_before = get_process_ram_mb()
            run_onnx_inference(session_onnx_quant, batches[0])
            ram_after = get_process_ram_mb()
            ram_onnx_quant = max(0.0, ram_after - ram_before)

            latencies_onnx_quant = []
            for batch in batches:
                t_start = time.perf_counter()
                run_onnx_inference(session_onnx_quant, batch)
                latencies_onnx_quant.append(time.perf_counter() - t_start)

            avg_latency_onnx_quant = float(np.mean(latencies_onnx_quant) * 1000)
            throughput_onnx_quant = (
                1.0 / np.mean(latencies_onnx_quant) if latencies_onnx_quant else 0.0
            )

            results["ONNX_INT8"] = {
                "Latency (ms)": avg_latency_onnx_quant,
                "Throughput (QPS)": throughput_onnx_quant,
                "Disk Size (MB)": disk_onnx_quant,
                "RAM Usage (MB)": ram_onnx_quant,
                "Cold Start (s)": cold_start_onnx_quant,
            }

        # Save results
        save_metrics(results, self.metrics_dir / "optimization_benchmarks.json")
        logger.info("Benchmarking suite completed and logged.")
        return results


def main() -> None:
    """CLI entrypoint for running optimization exporter and benchmarks."""
    # Reconfigure output streams to handle emojis on Windows cmd safely
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Optimize and benchmark fine-tuned sequence classifier."
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
        help="Run benchmarks on a tiny subset of inputs.",
    )
    args = parser.parse_args()

    try:
        optimizer = ModelOptimizer(args.model_dir, smoke_run=args.smoke_run)
        optimizer.export_onnx()
        optimizer.quantize_pytorch()
        optimizer.quantize_onnx()
        optimizer.benchmark_models()
    except Exception as e:
        logger.exception("Model optimization process failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
