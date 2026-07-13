# SupportAI Classifier Performance Benchmarks

This document contains benchmark metrics comparing our candidate classification models on the test split of the dataset.

## System Specifications
- **Inference Mode**: CPU-only
- **Concurrency**: Single-thread (OMP_NUM_THREADS=1)

## Benchmark Metrics

| Model | Accuracy | ECE | Latency | Throughput | Memory (MB) | Disk Size (MB) | Cold Start |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear SVM** | 90.71% | 0.0791 | 0.66 ms | 1515.6 QPS | 0.1 MB | 3.3 MB | 0.144 s |
| **PyTorch FP32** | 91.86% | 0.0196 | 15.82 ms | 63.2 QPS | 166.3 MB | 255.6 MB | 0.017 s |
| **ONNX INT8** | 91.71% | 0.0196 | 11.39 ms | 87.8 QPS | 71.4 MB | 64.8 MB | 0.188 s |

## Metrics Definitions
- **Accuracy**: Overall classification accuracy on the test split labels.
- **ECE (Expected Calibration Error)**: Quantifies prediction calibration (closer to 0 is better).
- **Latency**: Mean inference time to tokenize and predict a single support ticket.
- **Throughput**: Calculated queries processed per second (QPS).
- **Memory**: Resident Set Size (RSS) RAM increase recorded post-warmup.
- **Disk Size**: Sum of serialized weights and metadata.
- **Cold Start**: Total load time from filesystem to ready-to-infer state.

