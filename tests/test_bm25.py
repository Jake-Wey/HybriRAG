"""Tests for the BM25 sparse retrieval module."""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))
sys.path.append(src_root)

import pytest

from hybrirag.retriever import SparseRetriever

@pytest.fixture
def sparse() -> SparseRetriever:
    return SparseRetriever()


@pytest.fixture
def doc_ids() -> list[int]:
    return list(range(8))


@pytest.fixture
def corpus() -> list[str]:
    return [
        "Retrieval-augmented generation combines retrieval with language models.",
        "BM25 is a probabilistic ranking function used in information retrieval.",
        "Dense retrieval uses vector embeddings to find semantically similar documents.",
        "Sparse retrieval relies on term frequency and inverse document frequency.",
        "Hybrid search merges results from dense and sparse retrievers using RRF.",
        "The cross-encoder reranks candidate documents with an attention-based model.",
        "FAISS is a library for efficient similarity search and clustering of vectors.",
        "Sentence transformers produce high-quality embeddings for semantic search.",
    ]

class TestAddDocuments:
    """
    Verify that documents can be added to the BM25 index.
    """

    def test_add_documents_succeeds(self, sparse: SparseRetriever, doc_ids: list[int], corpus: list[str]):
        """Adding documents should not raise."""

        sparse.add_documents(doc_ids, corpus)

    def test_add_empty_lists(self, sparse: SparseRetriever):
        """Adding empty lists should not raise."""

        sparse.add_documents([], [])

    def test_add_documents_incremental(self, sparse: SparseRetriever, doc_ids: list[int], corpus: list[str]):
        """Documents can be added in multiple batches."""

        sparse.add_documents(doc_ids[:3], corpus[:3])
        sparse.add_documents(doc_ids[3:], corpus[3:])

        # After both adds, all docs should be queryable
        results = sparse.query("retrieval", top_k=10)
        assert len(results) > 0

    def test_add_mismatched_lengths_raises(self, sparse: SparseRetriever):
        """Passing doc_ids and texts of different lengths should raise."""

        with pytest.raises(Exception):
            sparse.add_documents([0, 1], ["only one text"])

class TestQueryTopK:
    """
    Verify that querying returns the correct number of results.
    """

    def test_query_returns_top_k(self, sparse: SparseRetriever, doc_ids: list[int], corpus: list[str]):
        sparse.add_documents(doc_ids, corpus)
        results = sparse.query("BM25 ranking", top_k=3)
        assert len(results) <= 3
        assert len(results) > 0

    def test_query_top_k_greater_than_corpus(self, sparse: SparseRetriever):
        """When top_k exceeds the number of indexed docs, return all matches."""

        sparse.add_documents([0], ["Only one document here."])
        results = sparse.query("document", top_k=10)
        assert len(results) == 1

    def test_query_returns_tuples(self, sparse: SparseRetriever, doc_ids: list[int], corpus: list[str]):
        """Each result should be a (doc_id, score) tuple."""

        sparse.add_documents(doc_ids, corpus)
        results = sparse.query("retrieval", top_k=3)
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            doc_id, score = result
            assert isinstance(doc_id, int)
            assert isinstance(score, float)

    def test_query_empty_index(self, sparse: SparseRetriever):
        """Querying an empty index should return an empty list."""

        results = sparse.query("anything", top_k=5)
        assert results == []

class TestBM25Scoring:
    """
    Verify that BM25 scoring produces sensible rankings.
    """

    def test_relevant_docs_score_higher(self, sparse: SparseRetriever):
        """A document containing the query term should rank above one that doesn't."""

        doc_ids = [0, 1, 2]
        docs = [
            "Python is a popular programming language for machine learning.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require large datasets for training.",
        ]
        sparse.add_documents(doc_ids, docs)
        results = sparse.query("machine learning", top_k=3)

        # The first and third documents should rank higher than the second
        top_ids = [r[0] for r in results]
        assert 1 not in top_ids[:1]

    def test_scores_are_non_negative(self, sparse: SparseRetriever, doc_ids: list[int], corpus: list[str]):
        sparse.add_documents(doc_ids, corpus)
        results = sparse.query("retrieval", top_k=5)
        for _, score in results:
            assert score >= 0.0

    def test_exact_term_match_ranks_high(self, sparse: SparseRetriever):
        """A document with an exact match of a rare term should rank first."""

        doc_ids = [0, 1, 2]
        docs = [
            "General text about many topics without specific keywords.",
            "This document mentions the rareterm xyzzy exactly.",
            "Another generic document with common words.",
        ]
        sparse.add_documents(doc_ids, docs)
        results = sparse.query("xyzzy", top_k=3)
        assert results[0][0] == 1
