# SupportAI Technical Interview Guide

This guide compiles engineering rationales, architectural design decisions, and deep-dive answers to common questions about the SupportAI project repository. It is designed to prepare you for technical reviews and interviews.

---

## 1. Core Architectural & Modeling Decisions

### Why did you choose DistilBERT over BERT?
- **Footprint**: DistilBERT has **40% fewer parameters** (66M vs 110M) than BERT-base.
- **Latency & Throughput**: It is **60% faster**, achieving much higher throughput (e.g., 63.2 QPS vs ~35 QPS on standard CPUs).
- **Accuracy Retention**: DistilBERT retains **97% of BERT's language comprehension capability** via knowledge distillation. For intent classification (Banking77 split), DistilBERT achieves a high test accuracy (**91.86%**), making the BERT overhead unjustifiable.
- **Resource Constraints**: It fits comfortably on commodity CPU hardware with a low memory profile (~166 MB RAM post-warmup), eliminating the need for expensive dedicated GPUs.

### Why compare SVM and DistilBERT?
- **Rule of Parsimony**: Always start with the simplest baseline. Linear SVM using TF-IDF text representations represents a classical, deterministic baseline that trains in seconds (~4.19s).
- **Informed Trade-Off**: SVM achieves **90.71% test accuracy** with a sub-millisecond inference latency (0.66 ms) and an incredibly small model footprint (3.3 MB). 
- **Business Rationale**: Comparing the two allows us to justify the complexity and cost of the deep learning model. DistilBERT gains **~1.15% absolute accuracy** over SVM, which might be critical for reducing downstream human escalation costs, but it requires **~80x more disk space** and **~24x higher latency**. Having both models allows the system to support a fallback mode or low-resource deployments.

### Why calibrate probabilities?
- **Overconfidence**: Modern deep neural networks are notoriously overconfident. They tend to output probabilities close to 1.0 or 0.0 even for incorrect predictions, because their training objective (cross-entropy) drives logits to infinity to minimize training loss.
- **Routing Reliability**: Our 3-tier routing engine relies on confidence thresholds (e.g., `high_confidence_threshold: 0.90` and `low_confidence_threshold: 0.60`). If the model's confidence scores are uncalibrated, a ticket might be auto-routed (Tier 1) based on a false 95% confidence, when its actual probability of being correct is closer to 40%, resulting in incorrect automated actions.
- **Decision Calibration**: Calibration techniques (like Temperature Scaling, $T = 1.1939$) scale the logits before softmax to align the predicted confidence scores with the empirical accuracy.

### Why not fine-tune a larger LLM?
- **Operational Costs**: Fine-tuning and serving a 7B or 8B parameter LLM (e.g., Llama-3 or Mistral) for intent classification requires significant GPU VRAM (~16 GB minimum for inference, 40 GB+ for training).
- **Task Specificity**: Banking77 has 77 highly specific, narrow intent classes. Fine-tuning a massive autoregressive model to output exactly one of 77 tokens is inefficient compared to a dedicated classification head on a small encoder (DistilBERT).
- **Latency**: Even a quantized 8B LLM has a latency of 100ms+ per forward pass. Our classifier runs in **11.39 ms** on CPU using ONNX INT8, representing a 10x-50x latency advantage.

---

## 2. Optimization & Search

### Why use ONNX INT8?
- **Throughput Gains**: Quantizing weights from FP32 to dynamic INT8 reduces memory bandwidth bottlenecks. This boosts CPU throughput from **63.2 QPS (PyTorch FP32) → 87.8 QPS (ONNX INT8)**.
- **Memory Footprint**: Memory usage is cut by **57%** (from 166.3 MB post-warmup to **71.4 MB**), making the container suitable for low-cost, serverless deployments (like AWS Fargate or Google Cloud Run).
- **Storage Savings**: Disk footprint drops by **74.6%** (from 255.6 MB to **64.8 MB**), speeding up CI/CD registry pulls and container startup times.
- **Minimal Accuracy Loss**: Quantization introduces only **0.15% absolute accuracy regression** (91.86% → 91.71%), which is a highly acceptable trade-off for the hardware and latency benefits.

### Why use FAISS?
- **Search Efficiency**: FAISS (Facebook AI Similarity Search) is highly optimized in C++ for vector similarity queries. It scales far better than naive brute-force cosine similarity implemented in pure Python/NumPy.
- **Normalized Inner Product**: By normalizing our sentence embeddings to unit length (L2 norm = 1.0), the Inner Product (`faiss.IndexFlatIP`) becomes mathematically identical to **Cosine Similarity**, providing high-quality semantic retrieval at sub-millisecond speeds.
- **Scaling**: If the historical ticket database grows from 10k to 10M records, FAISS supports indexing structures (like HNSW or IVF) that keep search latency sub-linear ($O(\log N)$) instead of linear ($O(N)$).

---

## 3. Decision Pipeline & Logic

### Why a confidence threshold?
- **Hybrid System**: Standard systems are binary: they either use cheap static logic (often inaccurate) or expensive LLMs (slow and costly).
- **Cost-Performance Curve**: By introducing confidence thresholds:
  - **Tier 1 (Confidence $\ge 90\%$)**: Auto-routes instantly using the classifier. This handles **~70-80% of common traffic** at sub-millisecond speeds with virtually zero cost.
  - **Tier 2 (Confidence $60\% - 90\%$)**: Invokes retrieval and LLM drafting. This handles ambiguous cases where context is needed.
  - **Tier 3 (Confidence $< 60\%$)**: Immediately escalates to a human, preventing the system from hallucinating or misrouting highly complex queries.

### Why retrieve before using an LLM?
- **Grounding (RAG)**: LLMs do not have access to internal business-specific resolved ticket history. Retrieving the top-3 most similar resolved tickets from FAISS provides the LLM with concrete, verified context cases.
- **Hallucination Prevention**: Rather than letting the LLM invent a solution, we instruct it to draft a response using the exact resolutions from the retrieved cases.
- **Token Efficiency**: By retrieving only the top-3 highly relevant cases, we keep the prompt short and fit within the context window of small, efficient models (like Phi-3-mini or Qwen-0.5B).

---

## 4. Practical Engineering Challenges

### What were the biggest engineering challenges?
1. **Calibration Over-Correction**: While temperature scaling minimized Brier Score and ECE on the validation set, it regressed ECE on the test set split. This highlighted that the test split logits were already well-calibrated, and scaling introduced validation-specific bias. We resolved this by auditing the calibration split distribution.
2. **LLM VRAM OOMs**: Running `Phi-3-mini` (3.8B parameters) in full FP32 precision requires **~15.2 GB of memory**. Loading this on standard CPU development containers caused immediate OOM crashes. We addressed this by implementing a **configurable LLM fallback layer** that supports a lightweight Hugging Face testing model (`tiny-random-gpt2`), local Ollama REST endpoints, or quantized GGUF execution.
3. **ORM / ONNX Compatibility**: Exporting `DistilBERT` to ONNX required wrapping the model to return logits directly and clearing conflicting `value_info` metadata to bypass ONNX Runtime execution engine issues on dynamic input sizes.

### What would you improve with more time?
1. **Asynchronous API Routing**: Make the `/predict` endpoint fully asynchronous when executing Tier 2 LLM generation, preventing a single long-running generation request from blocking the FastAPI event loop.
2. **Hybrid Retrieval**: Combine dense semantic retrieval (FAISS + Sentence Transformers) with sparse lexical retrieval (BM25) to catch exact keyword matches (e.g., specific error codes) that dense embeddings occasionally miss.
3. **Platt/Isotonic Scaling Calibration**: Transition from simple single-parameter temperature scaling to Platt scaling or isotonic regression to handle non-monotonic calibration curves.
4. **CI/CD Smoke Testing**: Integrate mock datasets and light ONNX checks into the GitHub Action pipeline to verify neural model serving without downloading gigabytes of weights on every commit.
