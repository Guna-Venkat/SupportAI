"""
retrieval.py
============
Provides semantic retrieval utilities using MiniLM embeddings and FAISS index
to locate semantically similar support tickets.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

from src.data.dataset import load_and_preprocess_dataset
from src.utils.constants import OUTPUT_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)
console = Console()


def compute_recall_at_k(
    retrieved_indices: list[int], ground_truth_indices: list[int], k: int
) -> float:
    """Computes Recall@K metric."""
    if not ground_truth_indices:
        return 0.0
    retrieved_set = set(retrieved_indices[:k])
    gt_set = set(ground_truth_indices)
    hits = len(retrieved_set.intersection(gt_set))
    return hits / len(gt_set)


def compute_reciprocal_rank(retrieved_indices: list[int], ground_truth_indices: list[int]) -> float:
    """Computes Reciprocal Rank (RR) metric."""
    gt_set = set(ground_truth_indices)
    for rank_idx, doc_idx in enumerate(retrieved_indices, 1):
        if doc_idx in gt_set:
            return 1.0 / rank_idx
    return 0.0


def compute_average_precision(
    retrieved_indices: list[int], ground_truth_indices: list[int]
) -> float:
    """Computes Average Precision (AP) metric."""
    if not ground_truth_indices:
        return 0.0
    gt_set = set(ground_truth_indices)
    num_hits = 0
    sum_precisions = 0.0
    for rank_idx, doc_idx in enumerate(retrieved_indices, 1):
        if doc_idx in gt_set:
            num_hits += 1
            precision_at_rank = num_hits / rank_idx
            sum_precisions += precision_at_rank
    return sum_precisions / len(gt_set)


class SemanticRetriever:
    """Manages semantic search indexing and retrieval using SentenceTransformers and FAISS."""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        metric: str = "cosine",
        device: str | None = None,
    ) -> None:
        """Initialises the retriever.

        Args:
            embedding_model_name: SentenceTransformers model name.
            metric: Distance metric to use ('cosine' or 'l2').
            device: Device ('cuda', 'cpu', or None to auto-detect).
        """
        self.embedding_model_name = embedding_model_name
        self.metric = metric.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading sentence embedding model: %s", self.embedding_model_name)
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)

        # Dimension size of MiniLM embeddings (usually 384)
        if hasattr(self.embedding_model, "get_embedding_dimension"):
            self.dimension = self.embedding_model.get_embedding_dimension()
        else:
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()

        # FAISS index properties
        self.index: faiss.Index | None = None
        self.corpus: list[str] = []
        self.normalize_embeddings = self.metric == "cosine"

    def build_index(self, corpus: list[str]) -> None:
        """Constructs a FAISS index from a corpus of texts.

        Args:
            corpus: List of text strings to index.
        """
        self.corpus = list(corpus)
        if not self.corpus:
            raise ValueError("Corpus cannot be empty.")

        logger.info("Encoding corpus of %d texts...", len(self.corpus))
        embeddings = self.embedding_model.encode(
            self.corpus,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        if self.normalize_embeddings:
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

        self.index.add(embeddings)
        logger.info("FAISS index constructed successfully.")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Searches for top_k semantically similar texts.

        Args:
            query: Query string.
            top_k: Number of results to retrieve.

        Returns:
            List of dictionaries containing text, score, and index.
        """
        if self.index is None:
            raise ValueError("Index has not been built or loaded.")

        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        if self.normalize_embeddings:
            faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank in range(top_k):
            idx = int(indices[0][rank])
            if idx == -1:
                continue
            score = float(distances[0][rank])
            results.append(
                {
                    "rank": rank + 1,
                    "index": idx,
                    "score": score,
                    "text": self.corpus[idx],
                }
            )
        return results

    def evaluate_retrieval(
        self,
        queries: list[str],
        ground_truth_indices: list[list[int]],
        k: int = 5,
    ) -> dict[str, float]:
        """Evaluates retrieval quality using Recall@K, MRR, and MAP.

        Args:
            queries: List of validation query strings.
            ground_truth_indices: List of lists containing indices of relevant corpus docs.
            k: Top-K evaluation cutoff.

        Returns:
            A dictionary containing evaluation metrics.
        """
        if self.index is None:
            raise ValueError("Index has not been built or loaded.")

        recalls = []
        rrs = []
        aps = []

        query_embeddings = self.embedding_model.encode(
            queries,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        if self.normalize_embeddings:
            faiss.normalize_L2(query_embeddings)

        _distances, indices = self.index.search(query_embeddings, k)

        for i in range(len(queries)):
            retrieved = indices[i].tolist()
            gt = ground_truth_indices[i]

            recalls.append(compute_recall_at_k(retrieved, gt, k))
            rrs.append(compute_reciprocal_rank(retrieved, gt))
            aps.append(compute_average_precision(retrieved, gt))

        return {
            f"Recall@{k}": float(np.mean(recalls)),
            "MRR": float(np.mean(rrs)),
            "MAP": float(np.mean(aps)),
        }

    def save_index(self, directory: Path | str) -> None:
        """Saves FAISS index and corpus metadata to disk.

        Args:
            directory: Directory path to save artifacts.
        """
        if self.index is None:
            raise ValueError("No index to save.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))

        metadata = {
            "embedding_model_name": self.embedding_model_name,
            "metric": self.metric,
            "dimension": self.dimension,
            "normalize_embeddings": self.normalize_embeddings,
            "corpus": self.corpus,
        }
        with open(directory / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info("Saved FAISS index and metadata to %s", directory)

    def load_index(self, directory: Path | str) -> None:
        """Loads FAISS index and corpus metadata from disk.

        Args:
            directory: Directory path containing the saved artifacts.
        """
        directory = Path(directory)
        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Missing FAISS files in {directory}")

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        self.embedding_model_name = metadata["embedding_model_name"]
        self.metric = metadata["metric"]
        self.dimension = metadata["dimension"]
        self.normalize_embeddings = metadata["normalize_embeddings"]
        self.corpus = metadata["corpus"]

        logger.info("FAISS index loaded successfully from %s", directory)


def main() -> None:
    """CLI entrypoint for executing semantic retrieval search queries."""
    parser = argparse.ArgumentParser(description="Query semantic ticket retrieval database.")
    parser.add_argument("--query", type=str, required=True, help="Query string to search.")
    parser.add_argument(
        "--index_dir",
        type=str,
        default=str(OUTPUT_DIR / "retrieval_index"),
        help="Directory where index is saved.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of items to retrieve.")
    parser.add_argument(
        "--build_from_dataset",
        action="store_true",
        help="If index is missing, builds it from local test dataset parquet.",
    )
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    retriever = SemanticRetriever()

    if not index_dir.exists():
        if args.build_from_dataset:
            console.print(
                "[yellow]Index directory not found. Building index from dataset...[/yellow]"
            )
            splits = load_and_preprocess_dataset()
            corpus = splits["test"]["text"].tolist()
            retriever.build_index(corpus)
            retriever.save_index(index_dir)
        else:
            console.print(
                f"[bold red]Error: Index directory {index_dir} not found. "
                "Run with --build_from_dataset to construct it.[/bold red]"
            )
            return
    else:
        retriever.load_index(index_dir)

    results = retriever.retrieve(args.query, top_k=args.top_k)

    console.print()
    console.print(f"[bold cyan]Semantic Search Results for query:[/bold cyan] '{args.query}'")
    console.print()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="center")
    table.add_column("Index", justify="center")
    table.add_column("Similarity Score", justify="right")
    table.add_column("Text / Content", justify="left")

    for res in results:
        # Cosine similarity scores close to 1.0 indicate high similarity
        score_val = res["score"]
        score_color = "green" if score_val > 0.6 else "yellow"
        table.add_row(
            str(res["rank"]),
            str(res["index"]),
            f"[{score_color}]{score_val:.4f}[/{score_color}]",
            res["text"][:100] + "..." if len(res["text"]) > 100 else res["text"],
        )

    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
