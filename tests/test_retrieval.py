"""
test_retrieval.py
=================
Unit tests for Phase 11: Semantic Retrieval.
"""

from pathlib import Path

from src.models.transformer.retrieval import (
    SemanticRetriever,
    compute_average_precision,
    compute_recall_at_k,
    compute_reciprocal_rank,
)


def test_metrics_calculation() -> None:
    # Retrieved indices: [0, 1, 2, 3, 4]
    # Ground truth indices: [2, 4]
    retrieved = [0, 1, 2, 3, 4]
    gt = [2, 4]

    # Recall@3: in top 3, only 2 is retrieved. So 1 / 2 = 0.5
    assert compute_recall_at_k(retrieved, gt, k=3) == 0.5
    # Recall@5: 2 and 4 are retrieved. So 2 / 2 = 1.0
    assert compute_recall_at_k(retrieved, gt, k=5) == 1.0

    # Reciprocal Rank: first hit is at index 2 (rank 3). So 1/3
    assert compute_reciprocal_rank(retrieved, gt) == 1.0 / 3.0

    # Average Precision:
    # hit 1: rank 3 (precision: 1/3)
    # hit 2: rank 5 (precision: 2/5)
    # AP = ((1/3) + (2/5)) / 2 = (5/15 + 6/15) / 2 = 11/30
    assert abs(compute_average_precision(retrieved, gt) - 11.0 / 30.0) < 1e-6


def test_semantic_retriever_pipeline(tmp_path: Path) -> None:
    corpus = [
        "How do I reset my account password?",
        "My laptop screen is completely black and won't turn on.",
        "I need help setting up billing details.",
        "Unable to log into the database server.",
    ]

    retriever = SemanticRetriever(metric="cosine")
    retriever.build_index(corpus)

    assert retriever.index is not None
    assert len(retriever.corpus) == 4

    # Test similarity retrieval
    results = retriever.retrieve("reset password", top_k=2)
    assert len(results) == 2
    assert results[0]["rank"] == 1
    assert "password" in results[0]["text"]

    # Test serialization
    index_dir = tmp_path / "retrieval_index"
    retriever.save_index(index_dir)

    new_retriever = SemanticRetriever(metric="cosine")
    new_retriever.load_index(index_dir)
    assert len(new_retriever.corpus) == 4

    new_results = new_retriever.retrieve("reset password", top_k=2)
    assert new_results[0]["text"] == results[0]["text"]

    # Test evaluation
    queries = ["password problem", "broken laptop screen"]
    ground_truth = [[0], [1]]
    eval_metrics = new_retriever.evaluate_retrieval(queries, ground_truth, k=2)

    assert "Recall@2" in eval_metrics
    assert "MRR" in eval_metrics
    assert "MAP" in eval_metrics
    assert eval_metrics["Recall@2"] == 1.0
    assert eval_metrics["MRR"] == 1.0
