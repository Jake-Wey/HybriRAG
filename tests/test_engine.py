"""Tests for the HybriRAGEngine end-to-end pipeline."""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))
sys.path.append(src_root)
from unittest.mock import MagicMock, patch

import pytest

from hybrirag.engine import HybriRAGEngine

@pytest.fixture
def sample_documents() -> list[dict]:
    """Return sample documents for engine tests."""

    return [
        {"id": "doc1", "text": "Retrieval-augmented generation improves model accuracy by grounding responses in external knowledge."},
        {"id": "doc2", "text": "BM25 is a bag-of-words retrieval function based on term frequency and inverse document frequency."},
        {"id": "doc3", "text": "Dense retrieval encodes queries and documents as vectors and finds nearest neighbors in embedding space."},
        {"id": "doc4", "text": "Hybrid retrieval combines sparse and dense methods, using RRF to fuse the ranked result lists."},
        {"id": "doc5", "text": "Cross-encoder reranking applies pairwise attention between query and document for precise relevance scoring."}
    ]


@pytest.fixture
def mock_engine():
    """Return a HybriRAGEngine with all sub-components mocked."""

    with patch("hybrirag.engine.DenseRetriever") as MockDense, \
         patch("hybrirag.engine.SparseRetriever") as MockSparse, \
         patch("hybrirag.engine.CrossEncoderReranker") as MockReranker, \
         patch("hybrirag.engine.SemanticChunker") as MockChunker, \
         patch("hybrirag.engine.QueryRouter") as MockRouter:

        # Configure chunker mock
        mock_chunker = MagicMock()
        mock_chunker.chunk.side_effect = lambda text: [text] if text else []
        MockChunker.return_value = mock_chunker

        # Configure dense retriever mock
        mock_dense = MagicMock()
        mock_dense.search.return_value = [
            (1, 0.9), (3, 0.8)
        ]
        MockDense.return_value = mock_dense

        # Configure sparse retriever mock
        mock_sparse = MagicMock()
        mock_sparse.query.return_value = [
            (1, 12.5), (4, 10.2)
        ]
        MockSparse.return_value = mock_sparse

        # Configure reranker mock
        mock_reranker_inst = MagicMock()
        mock_reranker_inst.rerank.return_value = [
            {"doc_id": 1, "text": "RAG improves accuracy.", "score": 0.98},
            {"doc_id": 4, "text": "Hybrid retrieval fusion.", "score": 0.91},
            {"doc_id": 2, "text": "BM25 ranking algorithm.", "score": 0.85},
        ]
        MockReranker.return_value = mock_reranker_inst

        # Configure router mock
        mock_router = MagicMock()
        mock_router.needs_retrieval.return_value = True
        MockRouter.return_value = mock_router

        engine = HybriRAGEngine(
            embedding_model="mock-embedder",
            rerank_model="mock-reranker",
        )

        mock_embedder = MagicMock()
        # Attach mocks for direct access in tests
        engine._dense_retriever = mock_dense
        engine._sparse_retriever = mock_sparse
        engine._reranker = mock_reranker_inst
        engine.router = mock_router
        engine._embedder = mock_embedder
        engine.chunker = mock_chunker
        return engine

class TestAddDocuments:
    """
    Verify that documents are added to all sub-indices.
    """

    def test_add_documents_calls_dense(self, mock_engine, sample_documents):
        mock_engine.add_documents(sample_documents)
        mock_engine._dense_retriever.add.assert_called_once()

    def test_add_documents_calls_sparse(self, mock_engine, sample_documents):
        mock_engine.add_documents(sample_documents)
        mock_engine._sparse_retriever.add_documents.assert_called_once()

    def test_add_documents_calls_chunker(self, mock_engine, sample_documents):
        mock_engine.add_documents(sample_documents)

        # Chunker should be called for each document
        assert mock_engine.chunker.chunk.call_count == len(sample_documents)

    def test_add_empty_list(self, mock_engine):
        mock_engine.add_documents([])
        mock_engine._dense_retriever.add.assert_not_called()
        mock_engine._sparse_retriever.add_documents.assert_not_called()

class TestRetrievePipeline:
    """
    Verify the full retrieve pipeline: dense -> sparse -> fuse -> rerank.
    """

    def test_retrieve_returns_results(self, mock_engine, sample_documents):
        mock_engine.add_documents(sample_documents)
        results = mock_engine.retrieve("What is hybrid retrieval?", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_retrieve_calls_dense_search(self, mock_engine):
        mock_engine.retrieve("What is hybrid retrieval?", top_k=3)
        mock_engine._dense_retriever.search.assert_called_once()

    def test_retrieve_calls_sparse_query(self, mock_engine):
        mock_engine.retrieve("What is hybrid retrieval?", top_k=3)
        mock_engine._sparse_retriever.query.assert_called_once()

    def test_retrieve_calls_reranker(self, mock_engine, sample_documents):
        mock_engine.add_documents(sample_documents)
        mock_engine.retrieve("What is hybrid retrieval?", top_k=3)
        mock_engine._reranker.rerank.assert_called_once()

    def test_retrieve_respects_top_k(self, mock_engine, sample_documents):
        mock_engine.add_documents(sample_documents)
        _ = mock_engine.retrieve("What is hybrid retrieval?", top_k=2)

        # Reranker was called with top_k=2
        call_args = mock_engine._reranker.rerank.call_args
        assert call_args.kwargs.get("top_k", call_args[1].get("top_k")) == 2

    def test_retrieve_result_format(self, mock_engine):
        results = mock_engine.retrieve("What is hybrid retrieval?", top_k=3)
        for result in results:
            assert "id" in result or hasattr(result, "id")
            assert "score" in result or hasattr(result, "score")

class TestQueryRouting:
    """
    Verify that the query router controls whether retrieval is performed.
    """

    def test_routing_needs_retrieval(self, mock_engine, sample_documents):
        """When the router says retrieval is needed, retrieve should run."""

        mock_engine.router.needs_retrieval.return_value = True
        mock_engine.add_documents(sample_documents)
        results = mock_engine.retrieve("What is BM25?")
        assert len(results) > 0
        mock_engine._dense_retriever.search.assert_called()

    def test_routing_skips_retrieval(self, mock_engine):
        """When the router says no retrieval is needed, it should skip search."""

        mock_engine.router.needs_retrieval.return_value = False
        results = mock_engine.retrieve("Hello, how are you?")

        # Dense and sparse search should not be called
        mock_engine._dense_retriever.search.assert_not_called()
        mock_engine._sparse_retriever.query.assert_not_called()
        assert results == []

    def test_routing_with_low_relevance(self, mock_engine, sample_documents):
        """When reranker scores are all very low, results should indicate low relevance."""

        mock_engine._reranker.rerank.return_value = [
            {"doc_id": 1, "text": "Low relevance.", "score": 0.05},
        ]
        mock_engine.add_documents(sample_documents)
        results = mock_engine.retrieve("completely unrelated query", top_k=3)

        # Results are returned but with low scores
        assert len(results) > 0
        assert results[0]["score"] < 0.1
