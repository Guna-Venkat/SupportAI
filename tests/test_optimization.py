"""
test_optimization.py
====================
Unit tests for Phase 9: ONNX and Quantization Optimization.
"""

from src.models.transformer.optimization import ModelOptimizer
from src.utils.constants import OUTPUT_DIR


def test_optimization_runner_smoke_pipeline(tmp_path) -> None:
    best_model_dir = OUTPUT_DIR / "models" / "best_model"

    # Initialize optimizer runner
    optimizer = ModelOptimizer(best_model_dir, smoke_run=True)

    # 1. Export ONNX
    onnx_path = optimizer.export_onnx()
    assert onnx_path.exists()

    # 2. Quantize PyTorch
    pytorch_quant_path = optimizer.quantize_pytorch()
    assert pytorch_quant_path.exists()

    # 3. Quantize ONNX
    onnx_quant_path = optimizer.quantize_onnx()
    assert onnx_quant_path.exists()

    # 4. Benchmarking
    results = optimizer.benchmark_models()

    assert "PyTorch_FP32" in results
    assert "PyTorch_INT8" in results
    assert "ONNX_FP32" in results
    assert "ONNX_INT8" in results

    # Verify JSON artifact was saved
    assert (OUTPUT_DIR / "metrics" / "optimization_benchmarks.json").exists()
