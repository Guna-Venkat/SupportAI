"""
benchmark.py
============
Evaluates Accuracy, ECE, Latency, Throughput, Memory, Model size, and Cold start
for SupportAI intent classification models.
Generates the markdown benchmark report automatically.
"""

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set cpu threads to 1 for standard single-thread benchmarking
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

DATA_PATH = Path("data/customer_support_tickets_v1/val.parquet")
MODELS_DIR = Path("outputs/models")
REPORT_PATH = Path("docs/benchmark_results.md")


def get_file_size_mb(path: Path) -> float:
    """Returns size of a file or directory in megabytes."""
    if not path.exists():
        return 0.0
    if path.is_file():
        return os.path.getsize(path) / (1024 * 1024)
    else:
        return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / (1024 * 1024)


def get_process_ram_mb() -> float:
    """Returns the RSS memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE) for multi-class classification."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(probs)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            ece += (bin_size / n_samples) * np.abs(accuracy_in_bin - confidence_in_bin)

    return ece


def main():
    print("=" * 60)
    print("SupportAI Unified Model Benchmarking Suite")
    print("=" * 60)

    # 1. Load validation dataset
    if not DATA_PATH.exists():
        print(f"Error: Validation dataset not found at {DATA_PATH}")
        return

    print(f"Loading validation dataset from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    texts = df["text"].tolist()

    # Load label mapping to convert string labels to IDs
    encoder_path = MODELS_DIR / "label_encoder.json"
    if not encoder_path.exists():
        print(f"Error: Label encoder not found at {encoder_path}")
        return

    with open(encoder_path, encoding="utf-8") as f:
        encoder_data = json.load(f)
    label_to_id = encoder_data["label_to_id"]

    first_val = df["label"].iloc[0]
    if isinstance(first_val, (int, np.integer)) or (
        isinstance(first_val, str) and first_val.isdigit()
    ):
        true_labels = np.array(df["label"].astype(int))
    else:
        true_labels = np.array([label_to_id[lbl] for lbl in df["label"]])

    print(f"Loaded {len(texts)} validation samples.")

    # Results dictionary
    results = {}

    # --- SVM Pipeline Benchmarking ---
    svm_path = MODELS_DIR / "linear_svm_pipeline.joblib"
    if svm_path.exists():
        print("\nBenchmarking Linear SVM Pipeline...")
        t0 = time.perf_counter()
        svm_model = joblib.load(svm_path)
        cold_start = time.perf_counter() - t0
        disk_size = get_file_size_mb(svm_path)

        ram_before = get_process_ram_mb()
        # Warmup
        has_predict_proba = hasattr(svm_model, "predict_proba")
        if has_predict_proba:
            try:
                svm_model.predict_proba([texts[0]])
            except AttributeError:
                has_predict_proba = False

        if not has_predict_proba:
            svm_model.decision_function([texts[0]])

        ram_after = get_process_ram_mb()
        ram_usage = max(0.0, ram_after - ram_before)

        latencies = []
        all_probs = []
        for txt in texts:
            ts = time.perf_counter()
            if has_predict_proba:
                probs = svm_model.predict_proba([txt])[0]
            else:
                scores = svm_model.decision_function([txt])
                exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                probs = (exp_scores / np.sum(exp_scores, axis=-1, keepdims=True))[0]
            latencies.append(time.perf_counter() - ts)
            all_probs.append(probs)

        all_probs = np.array(all_probs)
        predictions = np.argmax(all_probs, axis=1)
        accuracy = np.mean(predictions == true_labels)
        ece = calculate_ece(all_probs, true_labels)
        avg_latency = np.mean(latencies) * 1000  # ms
        throughput = 1.0 / np.mean(latencies)

        results["Linear SVM"] = {
            "Accuracy": f"{accuracy:.2%}",
            "ECE": f"{ece:.4f}",
            "Latency (ms)": f"{avg_latency:.2f} ms",
            "Throughput": f"{throughput:.1f} QPS",
            "Memory (MB)": f"{ram_usage:.1f} MB",
            "Size (MB)": f"{disk_size:.1f} MB",
            "Cold Start": f"{cold_start:.3f} s",
        }
    else:
        print("SVM model not found. Skipping SVM.")

    # --- PyTorch FP32 Benchmarking ---
    pt_path = MODELS_DIR / "best_model"
    if pt_path.exists():
        print("\nBenchmarking PyTorch FP32 (DistilBERT)...")
        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(str(pt_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(pt_path))
        model.eval()
        cold_start = time.perf_counter() - t0
        disk_size = get_file_size_mb(pt_path)

        ram_before = get_process_ram_mb()
        # Warmup
        inputs = tokenizer(
            [texts[0]], padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            model(**inputs)
        ram_after = get_process_ram_mb()
        ram_usage = max(0.0, ram_after - ram_before)

        latencies = []
        all_probs = []
        for txt in texts:
            ts = time.perf_counter()
            inputs = tokenizer(
                [txt], padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            latencies.append(time.perf_counter() - ts)
            all_probs.append(probs)

        all_probs = np.array(all_probs)
        predictions = np.argmax(all_probs, axis=1)
        accuracy = np.mean(predictions == true_labels)
        ece = calculate_ece(all_probs, true_labels)
        avg_latency = np.mean(latencies) * 1000  # ms
        throughput = 1.0 / np.mean(latencies)

        results["PyTorch FP32"] = {
            "Accuracy": f"{accuracy:.2%}",
            "ECE": f"{ece:.4f}",
            "Latency (ms)": f"{avg_latency:.2f} ms",
            "Throughput": f"{throughput:.1f} QPS",
            "Memory (MB)": f"{ram_usage:.1f} MB",
            "Size (MB)": f"{disk_size:.1f} MB",
            "Cold Start": f"{cold_start:.3f} s",
        }
    else:
        print("PyTorch FP32 model directory not found. Skipping.")

    # --- ONNX INT8 Benchmarking ---
    onnx_quant_path = MODELS_DIR / "model_quant.onnx"
    if onnx_quant_path.exists():
        print("\nBenchmarking ONNX INT8 Quantized...")
        import onnxruntime as ort

        t0 = time.perf_counter()
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(onnx_quant_path), sess_options)
        tokenizer = AutoTokenizer.from_pretrained(str(pt_path))
        cold_start = time.perf_counter() - t0
        disk_size = get_file_size_mb(onnx_quant_path)

        ram_before = get_process_ram_mb()
        # Warmup
        inputs = tokenizer(
            [texts[0]], padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        session.run(
            None,
            {
                "input_ids": inputs["input_ids"].numpy().astype(np.int64),
                "attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
            },
        )
        ram_after = get_process_ram_mb()
        ram_usage = max(0.0, ram_after - ram_before)

        latencies = []
        all_probs = []
        for txt in texts:
            ts = time.perf_counter()
            inputs = tokenizer(
                [txt], padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            logits = session.run(
                None,
                {
                    "input_ids": inputs["input_ids"].numpy().astype(np.int64),
                    "attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
                },
            )[0]
            # Softmax calculation
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            latencies.append(time.perf_counter() - ts)
            all_probs.append(probs[0])

        all_probs = np.array(all_probs)
        predictions = np.argmax(all_probs, axis=1)
        accuracy = np.mean(predictions == true_labels)
        ece = calculate_ece(all_probs, true_labels)
        avg_latency = np.mean(latencies) * 1000  # ms
        throughput = 1.0 / np.mean(latencies)

        results["ONNX INT8"] = {
            "Accuracy": f"{accuracy:.2%}",
            "ECE": f"{ece:.4f}",
            "Latency (ms)": f"{avg_latency:.2f} ms",
            "Throughput": f"{throughput:.1f} QPS",
            "Memory (MB)": f"{ram_usage:.1f} MB",
            "Size (MB)": f"{disk_size:.1f} MB",
            "Cold Start": f"{cold_start:.3f} s",
        }
    else:
        print("ONNX Quantized model not found. Skipping ONNX INT8.")

    # --- Print Summary Table to Console ---
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 80)

    headers = [
        "Model",
        "Accuracy",
        "ECE",
        "Latency (ms)",
        "Throughput (QPS)",
        "Memory (MB)",
        "Size (MB)",
        "Cold Start (s)",
    ]
    print(f"{' | '.join(headers)}")
    print("-" * 120)
    for model_name, metrics in results.items():
        row = [
            model_name,
            metrics["Accuracy"],
            metrics["ECE"],
            metrics["Latency (ms)"],
            metrics["Throughput"],
            metrics["Memory (MB)"],
            metrics["Size (MB)"],
            metrics["Cold Start"],
        ]
        print(" | ".join(row))

    # --- Write Markdown Report to docs/benchmark_results.md ---
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    md_content = """# SupportAI Classifier Performance Benchmarks

This document contains automatically generated benchmark metrics comparing our
candidate classification models on the validation split of the dataset.

## System Specifications
- **Inference Mode**: CPU-only
- **Concurrency**: Single-thread (OMP_NUM_THREADS=1)

## Benchmark Metrics

| Model | Accuracy | ECE | Latency | Throughput | Memory (MB) | Disk Size (MB) | Cold Start |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for model_name, metrics in results.items():
        md_content += f"| **{model_name}** | {metrics['Accuracy']} | {metrics['ECE']} | {metrics['Latency (ms)']} | {metrics['Throughput']} | {metrics['Memory (MB)']} | {metrics['Size (MB)']} | {metrics['Cold Start']} |\n"  # noqa: E501

    md_content += """
## Metrics Definitions
- **Accuracy**: Overall classification accuracy on the validation split labels.
- **ECE (Expected Calibration Error)**: Quantifies prediction calibration
  (closer to 0 is better).
- **Latency**: Mean inference time to tokenize and predict a single support ticket.
- **Throughput**: Calculated queries processed per second (QPS).
- **Memory**: Resident Set Size (RSS) RAM increase recorded post-warmup.
- **Disk Size**: Sum of serialized weights and metadata.
- **Cold Start**: Total load time from filesystem to ready-to-infer state.
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nBenchmark results successfully written to {REPORT_PATH}\n")


if __name__ == "__main__":
    main()
