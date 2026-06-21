"""Tests for the Cross-Encoder reranker module."""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))
sys.path.append(src_root)
from unittest.mock import MagicMock, patch

import pytest
import torch

from hybrirag.reranker import CrossEncoderReranker

@pytest.fixture
def candidate_ids() -> list[int]:
    return [1, 2, 3, 4, 5]

@pytest.fixture
def candidate_docs() -> list[str]:
    """Return candidate documents for reranking."""

    return [
        "Retrieval-augmented generation reduces hallucination.", 
        "BM25 is a term-based ranking algorithm.",
        "Dense retrieval uses neural embeddings for search.",
        "Cross-encoders perform pair-wise relevance scoring.",
        "FAISS enables fast approximate nearest neighbor search."
    ]


@pytest.fixture
def mock_reranker():
    """Return a CrossEncoderReranker with mocked model inference."""

    with patch("hybrirag.reranker.CrossEncoderReranker._load_model") as mock_load:
        reranker = CrossEncoderReranker(model_path="mock-reranker")

        # Simulate model predictions: return decreasing scores
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.95], [0.72], [0.88], [0.91], [0.45]])
        mock_model.return_value = mock_output
        mock_tokenizer = MagicMock()
        reranker._model = mock_model
        reranker._tokenizer = mock_tokenizer
        return reranker

class TestReranking:
    """
    Verify that reranking sorts documents by relevance score.
    """

    def test_rerank_returns_sorted(self, mock_reranker, candidate_docs,
                                   candidate_ids):
        """Reranked results should be sorted in descending score order."""

        results = mock_reranker.rerank(
            query="retrieval search",
            documents=candidate_docs,
            doc_ids=candidate_ids
        )
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_preserves_all_docs(self, mock_reranker, candidate_docs,
                                       candidate_ids):
        """Reranking should not drop any documents (unless top_k is set to le 5)."""

        results = mock_reranker.rerank(
            query="retrieval search",
            documents=candidate_docs,
            doc_ids=candidate_ids
        )
        assert len(results) == len(candidate_docs)

    def test_rerank_preserves_document_fields(self, mock_reranker, candidate_docs,
                                              candidate_ids):
        """Each reranked result should retain its original fields plus updated score."""

        results = mock_reranker.rerank(
            query="retrieval search",
            documents=candidate_docs,
            doc_ids=candidate_ids
        )

        for result in results:
            assert "doc_id" in result
            assert "text" in result
            assert "score" in result

    def test_rerank_empty_candidates(self, mock_reranker):
        """Reranking an empty list should return an empty list."""

        results = mock_reranker.rerank(query="test", documents=[], doc_ids=[])
        assert results == []

    def test_rerank_single_document(self, mock_reranker):
        """Reranking a single document should return it with an updated score."""

        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.99]])
        mock_reranker._model.return_value = mock_output
        docs = ["Only document."]
        results = mock_reranker.rerank(query="test", documents=docs, doc_ids=[1])
        assert len(results) == 1
        assert abs(results[0]["score"] - 0.99) < 1e-6

class TestTopKTruncation:
    """
    Verify that top_k limits the number of returned results.
    """

    def test_top_k_limits_results(self, mock_reranker, candidate_docs, candidate_ids):
        """When top_k=3, only 3 results should be returned."""

        results = mock_reranker.rerank(
            query="retrieval search",
            documents=candidate_docs,
            doc_ids=candidate_ids,
            top_k=3
        )
        assert len(results) == 3

    def test_top_k_greater_than_candidates(self, mock_reranker, candidate_docs,
                                           candidate_ids):
        """top_k larger than candidate count should return all candidates."""

        results = mock_reranker.rerank(
            query="retrieval search",
            documents=candidate_docs,
            doc_ids=candidate_ids,
            top_k=100
        )
        assert len(results) == len(candidate_docs)

    def test_top_k_one(self, mock_reranker, candidate_docs, candidate_ids):
        """top_k=1 should return only the single best document."""

        results = mock_reranker.rerank(
            query="retrieval search",
            documents=candidate_docs,
            doc_ids=candidate_ids,
            top_k=1,
        )
        assert len(results) == 1

    def test_top_k_results_are_highest_scored(self, mock_reranker, 
                                              candidate_docs, candidate_ids):
        """Returned results should be the top-k highest-scored documents."""

        results = mock_reranker.rerank(
            query="retrieval search",
            documents=candidate_docs,
            doc_ids=candidate_ids,
            top_k=3,
        )

        # All top-k results should have scores >= any omitted result
        all_results = mock_reranker.rerank(
            query="retrieval search",
            documents=candidate_docs,
            doc_ids=candidate_ids
        )
        min_top_k_score = min(r["score"] for r in results)
        omitted_scores = [r["score"] for r in all_results[len(results):]]
        for omitted_score in omitted_scores:
            assert min_top_k_score >= omitted_score
