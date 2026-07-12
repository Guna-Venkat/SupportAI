# SupportAI Classifier Performance Benchmarks

This document contains automatically generated benchmark metrics comparing our
candidate classification models on the validation split of the dataset.

## System Specifications
- **Inference Mode**: CPU-only
- **Concurrency**: Single-thread (OMP_NUM_THREADS=1)

## Benchmark Metrics

| Model | Accuracy | ECE | Latency | Throughput | Memory (MB) | Disk Size (MB) | Cold Start |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear SVM** | 88.24% | 0.7914 | 0.66 ms | 1515.6 QPS | 0.1 MB | 3.1 MB | 0.144 s |
| **PyTorch FP32** | 1.60% | 0.0004 | 27.48 ms | 36.4 QPS | 168.7 MB | 256.3 MB | 0.944 s |
| **ONNX INT8** | 1.60% | 0.0004 | 17.08 ms | 58.5 QPS | 1.0 MB | 64.9 MB | 0.433 s |

## Metrics Definitions
- **Accuracy**: Overall classification accuracy on the validation split labels.
- **ECE (Expected Calibration Error)**: Quantifies prediction calibration
  (closer to 0 is better).
- **Latency**: Mean inference time to tokenize and predict a single support ticket.
- **Throughput**: Calculated queries processed per second (QPS).
- **Memory**: Resident Set Size (RSS) RAM increase recorded post-warmup.
- **Disk Size**: Sum of serialized weights and metadata.
- **Cold Start**: Total load time from filesystem to ready-to-infer state.
