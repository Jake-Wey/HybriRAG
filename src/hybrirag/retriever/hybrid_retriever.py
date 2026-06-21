"""Hybrid retriever combining dense and sparse retrieval via Reciprocal Rank Fusion."""

import logging

import numpy as np

from hybrirag.retriever.dense_retriever import DenseRetriever
from hybrirag.retriever.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)

class HybridRetrieverError(Exception):
    """Raised when hybrid retrieval operations fail."""

class HybridRetriever:
    """Merge dense and sparse retrieval results using Reciprocal Rank Fusion."""

    RRF_K: int = 60 # RRF constant

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise HybridRetrieverError("Weights must be non-negative")
        
        logger.info(
            "HybridRetriever created (dense_w=%.2f, sparse_w=%.2f, rrf_k=%d)",
            self.dense_weight,
            self.sparse_weight,
            self.RRF_K,
        )

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 30,
        dense_top_k: int = 20,
        sparse_top_k: int = 20
    ) -> list[tuple[int, float]]:
        """
        Retrieve documents by merging dense and sparse results via RRF.

        Args:
            query: Raw query text (passed to the sparse retriever).
            query_embedding: Pre-computed query embedding vector (passed to the 
                dense retriever).
            top_k: Number of final merged results to return.
            dense_top_k: Number of candidates requested from the dense retriever.
            sparse_top_k: Number of candidates requested from the sparse retriever.

        Returns:
            (doc_id, rrf_score) pairs sorted by descending RRF score.
        """

        dense_results = self.dense_retriever.search(query_embedding, top_k=dense_top_k)
        logger.debug("Dense retrieval returned %d results.", len(dense_results))

        sparse_results = self.sparse_retriever.query(query, top_k=sparse_top_k)
        logger.debug("Sparse retrieval returned %d results.", len(sparse_results))

        rrf_scores: dict[int, float] = {}

        for rank_0, (doc_id, _) in enumerate(dense_results):
            rank = rank_0 + 1
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self.dense_weight / (
                self.RRF_K + rank
            )

        for rank_0, (doc_id, _) in enumerate(sparse_results):
            rank = rank_0 + 1
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self.sparse_weight / (
                self.RRF_K + rank
            )

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
