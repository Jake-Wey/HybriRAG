"""Tests for the retriever modules (Dense, Sparse, Hybrid)."""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))
sys.path.append(src_root)
from unittest.mock import MagicMock

import numpy as np
import pytest

from hybrirag.retriever import DenseRetriever, SparseRetriever, HybridRetriever

DIMENSION = 384  # bge-small-en-v1.5 embedding dimension


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Return mock L2-normalised embedding vectors (5 docs x DIMENSION dims)."""

    np.random.seed(99)
    vecs = np.random.randn(5, DIMENSION).astype("float32")

    # L2-normalise each row
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    return vecs

@pytest.fixture
def sample_doc_ids() -> list[int]:
    return [10, 20, 30, 40, 50]

class TestDenseRetriever:
    """
    Test DenseRetriever add and search operations.
    """

    def test_add_and_count(self, sample_embeddings, sample_doc_ids):
        """Documents should be indexed in FAISS after add."""

        retriever = DenseRetriever(dimension=DIMENSION)
        assert retriever.count == 0
        retriever.add(sample_embeddings, sample_doc_ids)
        assert retriever.count == 5

    def test_search_returns_top_k(self, sample_embeddings, sample_doc_ids):
        """Search should return at most top_k results."""

        retriever = DenseRetriever(dimension=DIMENSION)
        retriever.add(sample_embeddings, sample_doc_ids)

        # Use the first document's embedding as the query
        query_emb = sample_embeddings[0]
        results = retriever.search(query_emb, top_k=3)
        assert len(results) <= 3
        assert len(results) > 0

    def test_search_result_format(self, sample_embeddings, sample_doc_ids):
        """Each search result should be a (doc_id, score) tuple."""

        retriever = DenseRetriever(dimension=DIMENSION)
        retriever.add(sample_embeddings, sample_doc_ids)

        query_emb = sample_embeddings[0]
        results = retriever.search(query_emb, top_k=3)
        for doc_id, score in results:
            assert isinstance(doc_id, int)
            assert isinstance(score, float)

    def test_search_empty_index(self):
        """Searching an empty index should return an empty list."""

        retriever = DenseRetriever(dimension=DIMENSION)
        query = np.random.randn(DIMENSION).astype("float32")
        assert retriever.search(query, top_k=5) == []

    def test_self_match_ranks_first(self, sample_embeddings, sample_doc_ids):
        """A document's own embedding should be its own best match."""

        retriever = DenseRetriever(dimension=DIMENSION)
        retriever.add(sample_embeddings, sample_doc_ids)

        query_emb = sample_embeddings[2]  # doc_id=30
        results = retriever.search(query_emb, top_k=1)
        assert results[0][0] == 30

class TestSparseRetriever:
    """Test SparseRetriever (BM25-based) add and query operations."""

    def test_add_documents(self):
        retriever = SparseRetriever()
        retriever.add_documents([0, 1, 2], ["doc zero", "doc one", "doc two"])
        results = retriever.query("zero", top_k=3)
        assert len(results) > 0

    def test_query_returns_top_k(self):
        retriever = SparseRetriever()
        doc_ids = list(range(5))
        docs = [
            "Retrieval-augmented generation improves accuracy.",
            "BM25 uses term frequency for keyword matching.",
            "Dense retrieval encodes text into vector embeddings.",
            "Hybrid search combines dense and sparse methods.",
            "Cross-encoder models perform fine-grained reranking.",
        ]
        retriever.add_documents(doc_ids, docs)
        results = retriever.query("BM25 keyword", top_k=3)
        assert len(results) <= 3
        assert len(results) > 0

    def test_query_empty_index(self):
        """Querying an empty index should return no results."""

        retriever = SparseRetriever()
        results = retriever.query("anything", top_k=5)
        assert len(results) == 0

    def test_keyword_match_ranks_high(self):
        """Documents containing the query term should rank higher."""

        retriever = SparseRetriever()
        doc_ids = [0, 1, 2, 3, 4]
        docs = [
            "Retrieval-augmented generation improves accuracy.",
            "BM25 uses term frequency for keyword matching.",
            "Dense retrieval encodes text into vector embeddings.",
            "Hybrid search combines dense and sparse methods.",
            "Cross-encoder models perform fine-grained reranking.",
        ]
        retriever.add_documents(doc_ids, docs)
        results = retriever.query("BM25", top_k=5)

        # The BM25 document (doc_id=1) should rank near the top
        top_ids = [r[0] for r in results]
        assert 1 in top_ids[:2]

class TestHybridRetriever:
    """
    Test HybridRetriever RRF fusion of dense and sparse results.
    """

    def test_retrieve_returns_fused_results(self, sample_embeddings, sample_doc_ids):
        """Hybrid retrieve should return results from both paths."""

        dense = DenseRetriever(dimension=DIMENSION)
        dense.add(sample_embeddings, sample_doc_ids)

        sparse = SparseRetriever()
        sparse.add_documents(
            sample_doc_ids,
            [
                "RAG improves accuracy.",
                "BM25 ranking algorithm.",
                "Dense retrieval uses vectors.",
                "Hybrid retrieval fusion.",
                "Cross-encoder reranking.",
            ],
        )

        hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
        query_emb = sample_embeddings[0]
        results = hybrid.retrieve("retrieval search", query_emb, top_k=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_retrieve_result_format(self, sample_embeddings, sample_doc_ids):
        """Each result should be a (doc_id, rrf_score) tuple."""

        dense = DenseRetriever(dimension=DIMENSION)
        dense.add(sample_embeddings, sample_doc_ids)

        sparse = SparseRetriever()
        sparse.add_documents(
            sample_doc_ids,
            ["doc 0", "doc 1", "doc 2", "doc 3", "doc 4"],
        )

        hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
        query_emb = sample_embeddings[0]
        results = hybrid.retrieve("query", query_emb, top_k=5)
        for doc_id, score in results:
            assert isinstance(doc_id, int)
            assert isinstance(score, float)

    def test_rrf_fusion_merges_both_lists(self):
        """RRF should combine two ranked lists into a single fused ranking."""

        dense = MagicMock(spec=DenseRetriever)
        dense.search.return_value = [
            (1, 0.95),
            (2, 0.85),
            (3, 0.75),
        ]

        sparse = MagicMock(spec=SparseRetriever)
        sparse.query.return_value = [
            (2, 10.5),
            (4, 8.3),
            (1, 5.1),
        ]

        hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
        results = hybrid.retrieve("test query", np.zeros(DIMENSION, dtype="float32"), top_k=10)

        all_ids = {r[0] for r in results}
        assert all_ids == {1, 2, 3, 4}

        # doc 1 and 2 appear in both lists, so they should rank higher
        top_ids = [r[0] for r in results[:2]]
        assert 1 in top_ids or 2 in top_ids

    def test_rrf_fusion_empty_inputs(self):
        """RRF should handle empty result lists gracefully."""

        dense = MagicMock(spec=DenseRetriever)
        dense.search.return_value = []

        sparse = MagicMock(spec=SparseRetriever)
        sparse.query.return_value = []

        hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
        results = hybrid.retrieve("test", np.zeros(DIMENSION, dtype="float32"), top_k=10)
        assert results == []

    def test_rrf_fusion_one_empty(self):
        """RRF with one empty list should return results from the other list."""

        dense = MagicMock(spec=DenseRetriever)
        dense.search.return_value = [
            (1, 0.9),
            (2, 0.8),
        ]

        sparse = MagicMock(spec=SparseRetriever)
        sparse.query.return_value = []

        hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
        results = hybrid.retrieve("test", np.zeros(DIMENSION, dtype="float32"), top_k=10)
        assert len(results) == 2
