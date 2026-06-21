"""Sparse (BM25) retriever with optional C++ acceleration."""

import sys
import os
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import _cpp_bm25 # type: ignore

logger = logging.getLogger(__name__)

class SparseRetrieverError(Exception):
    """Raised when sparse retrieval operations fail."""

class SparseRetriever:
    """BM25-based sparse retriever"""

    def __init__(self) -> None:
        self._cpp_bm25 = _cpp_bm25.BM25Engine()

    def add_documents(self, doc_ids: list[int], texts: list[str]) -> None:
        """
        Index documents for BM25 retrieval.

        Args:
            doc_ids: Integer identifiers for each document.
            texts: Raw document strings.
        """

        if len(doc_ids) != len(texts):
            raise SparseRetrieverError("doc_ids and texts must have same length")
        
        self._cpp_bm25.add_documents(doc_ids, texts)

        logger.debug("Indexed %d documents in sparse retriever.", len(doc_ids))

    def query(self, query_text: str, top_k: int = 20) -> list[tuple[int, float]]:
        """
        Retrieve the top-k documents matching *query_text* via BM25.

        Args:
            query_text: The search query.
            top_k: Number of results to return.

        Returns:
            (doc_id, bm25_score) pairs sorted by descending score.
        """

        return self._cpp_bm25.query(query_text, top_k)    
