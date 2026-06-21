"""Main HybriRAG engine -- the top-level orchestrator."""

import logging
from typing import Any

from hybrirag.chunker import SemanticChunker
from hybrirag.embedder import Embedder
from hybrirag.reranker import CrossEncoderReranker
from hybrirag.retriever import DenseRetriever, HybridRetriever, SparseRetriever
from hybrirag.router import QueryRouter

handler = logging.FileHandler("app.log")
logger = logging.getLogger(__name__)
logger.addHandler(handler)

class HybriRAGEngineError(Exception):
    """Raised when the engine encounters a fatal error."""

class HybriRAGEngine:
    """Orchestrate the full RAG pipeline: chunk -> embed -> retrieve -> rerank."""

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        rerank_model: str = "BAAI/bge-reranker-base",
        device: str = "cuda",
        chunk_min_size: int = 50,
        chunk_max_size: int = 500,
        chunk_overlap: int = 1
    ) -> None:
        self.embedding_model_path = embedding_model
        self.rerank_model_path = rerank_model
        self.device = device

        self.chunker = SemanticChunker(
            min_chunk_size=chunk_min_size,
            max_chunk_size=chunk_max_size,
            overlap_sentences=chunk_overlap,
        )
        self.router = QueryRouter()

        self._embedder: Embedder | None = None
        self._dense_retriever: DenseRetriever | None = None
        self._sparse_retriever: SparseRetriever | None = None
        self._hybrid_retriever: HybridRetriever | None = None
        self._reranker: CrossEncoderReranker | None = None

        self._chunks: dict[int, dict] = {}
        self._next_chunk_id: int = 0

        logger.info(
            "HybriRAGEngine created (embed=%s, rerank=%s, device=%s)",
            embedding_model,
            rerank_model,
            device,
        )

    @property
    def embedder(self) -> Embedder:
        """Return the embedder, creating it on first access."""

        if self._embedder is None:
            self._embedder = Embedder(
                model_path=self.embedding_model_path, device=self.device
            )
            dim = self._embedder.dimension
            self._dense_retriever = DenseRetriever(dimension=dim)
        return self._embedder
    
    @property
    def dense_retriever(self) -> DenseRetriever:
        """Return the dense retriever, creating it on first access."""

        if self._dense_retriever is None:
            _ = self.embedder
        assert self._dense_retriever is not None
        return self._dense_retriever
    
    @property
    def sparse_retriever(self) -> SparseRetriever:
        """Return the sparse retriever, creating it on first access."""

        if self._sparse_retriever is None:
            self._sparse_retriever = SparseRetriever()
        return self._sparse_retriever
    
    @property
    def hybrid_retriever(self) -> HybridRetriever:
        """Return the hybrid retriever, creating it on first access."""

        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                dense_retriever=self.dense_retriever,
                sparse_retriever=self.sparse_retriever
            )
        return self._hybrid_retriever
    
    @property
    def reranker(self) -> CrossEncoderReranker:
        """Return the reranker, creating it on first access."""

        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                model_path=self.rerank_model_path, device=self.device
            )
        return self._reranker
    
    @property
    def chunk_count(self) -> int:
        """Number of chunks currently indexed."""

        return len(self._chunks)
    
    def add_documents(self, documents: list[str] | str) -> None:
        """
        Process and index one or more documents.

        Args:
            documents: A single document string or a list of document strings.
        """
        
        if isinstance(documents, str):
            documents = [documents]

        all_chunks: list[dict] = []
        all_ids: list[int] = []
        all_texts: list[str] = []

        for doc in documents:
            chunks = self.chunker.chunk(doc)
            for chunk in chunks:
                chunk_id = self._next_chunk_id
                self._next_chunk_id += 1

                chunk["chunk_id"] = chunk_id
                self._chunks[chunk_id] = chunk

                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_texts.append(chunk["text"])

        if not all_chunks:
            logger.warning("No chunks produced from %d document(s).", len(documents))
            return
        
        # --- Embed chunks ---
        logger.info("Embedding %d chunks ...", len(all_texts))
        embeddings = self.embedder.encode(all_texts, batch_size=32)

        # --- Add to dense index ---
        self.dense_retriever.add(embeddings, all_ids)

        # --- Add to sparse index ---
        self.sparse_retriever.add_documents(all_ids, all_texts)

        logger.info(
            "Indexed %d chunks from %d document(s). Total chunks: %d",
            len(all_chunks),
            len(documents),
            self.chunk_count
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Run the full retrieval pipeline: route -> embed -> hybrid retrieve -> rerank.

        Args:
            query: The search query.
            top_k: Number of final results to return after reranking.

        Returns:
            list[dict]: ach dict has keys chunk_id, score, text.
        """

        # 1. Routing -- skip retrieval for greetings / short queries.
        if not self.router.needs_retrieval(query):
            logger.info("Query does not need retrieval: '%s'", query[:80])
            return []
        
        # 2. Embed the query.
        query_embedding = self.embedder.encode_query(query)

        # 3. Hybrid retrieval.
        hybrid_top_k = max(top_k * 10, 30)
        candidates = self.hybrid_retriever.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=hybrid_top_k
        )
        if not candidates:
            logger.info("No candidates returned from hybrid retriever.")
            return []
        
        # 4. Check relevance.
        scores = [score for _, score in candidates]
        if not self.router.check_relevance(scores):
            logger.info("All candidate scores below relevance threshold.")
            return []
        
        # 5. Rerank.
        doc_ids = [cid for cid, _ in candidates]
        doc_texts = [
            self._chunks[cid]["text"] for cid in doc_ids if cid in self._chunks
        ]
        valid_ids = [cid for cid in doc_ids if cid in self._chunks]

        if not valid_ids:
            return []
        
        reranked = self.reranker.rerank(
            query=query,
            documents=doc_texts,
            doc_ids=valid_ids,
            top_k=top_k
        )

        # 6. Format results.
        results = []
        for item in reranked:
            chunk = self._chunks.get(item["doc_id"], {})
            results.append(
                {
                    "chunk_id": item["doc_id"],
                    "score": item["score"],
                    "text": item["text"],
                    "start": chunk.get("start"),
                    "end": chunk.get("end")
                }
            )

        return results
    
    def query(self, query: str, top_k: int = 3) -> dict[str, Any]:
        """
        Retrieve and format results for LLM consumption.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            dict
        """

        needs_retrieval = self.router.needs_retrieval(query)

        if not needs_retrieval:
            return {
                "results": [],
                "needs_retrieval": False,
                "query": query
            }
        
        results = self.retrieve(query, top_k=top_k)

        # Strip internal metadata for external consumption.
        clean_results = [
            {"chunk_id": r["chunk_id"], "score": r["score"], "text": r["text"]}
            for r in results
        ]

        return {
            "results": clean_results,
            "needs_retrieval": True,
            "query": query
        }
