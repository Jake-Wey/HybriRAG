"""FastAPI-based HTTP server for HybriRAG."""

from typing import Any
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from hybrirag.engine import HybriRAGEngine

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    """Schema for the ``/query`` endpoint request body."""

    query: str = Field(..., min_length=1, description="The search query")
    top_k: int = Field(default=3, ge=1, le=50, description="Number of results")

class QueryResponse(BaseModel):
    """Schema for the ``/query`` endpoint response."""

    results: list[dict[str, Any]] = Field(default_factory=list)
    needs_retrieval: bool = True

class DocumentAddResponse(BaseModel):
    """Schema for the ``/documents`` endpoint response."""
    status: str = "ok"
    chunks_indexed: int = 0

def create_app(engine: Any = None) -> FastAPI:
    """
    Build and return a :class:`FastAPI` application.
    
    Args:
        engine: A :class:`~hybrirag.engine.HybriRAGEngine` instance.  

    Returns:
        FastAPI: The configured application.
    """

    app = FastAPI(
        title="HybriRAG",
        version="0.1.0",
        description="Hybrid Retrieval-Augmented Generation Engine"
    )

    _engine_ref: list[Any] = [engine]

    def _get_engine() -> Any:
        """
        Lazily initialise the engine if not provided at construction.
        """

        if _engine_ref[0] is None:
            logger.info("Initialising default HybriRAGEngine ...")
            _engine_ref[0] = HybriRAGEngine()
        return _engine_ref[0]

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """
        Return service health status.
        """

        return {"status": "healthy"}
    
    @app.post("/documents", response_model=DocumentAddResponse)
    async def add_documents(
        file: UploadFile | None = File(default=None),
        texts: list[str] | None = None
    ) -> DocumentAddResponse:
        """
        Add documents to the index.
        """

        eng = _get_engine()

        document_texts: list[str] = []

        if file is not None:
            try:
                content = (await file.read()).decode("utf-8", errors="replace")

                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        document_texts.append(line)
            except Exception as e:
                logger.error("Failed to read uploaded file: %s", e)
                raise HTTPException(
                    status_code=400, detail=f"Failed to read file: {e}"
                ) from e
            
        if texts:
            document_texts.extend(t for t in texts if t.strip())

        if not document_texts:
            raise HTTPException(
                status_code=400, detail="No documents provided. Supply a file or texts list."
            )
        
        try:
            eng.add_documents(document_texts)
            chunks_indexed = eng.chunk_count if hasattr(eng, "chunk_count") \
                else len(document_texts)
        except Exception as e:
            logger.error("Document indexing failed: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Indexing failed: {e}"
            ) from e
        
        return DocumentAddResponse(status="ok", chunks_indexed=chunks_indexed)

    @app.post("/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest) -> QueryResponse:
        """
        Query the RAG pipeline.
        """

        eng = _get_engine()

        try:
            result = eng.query(query=request.query, top_k=request.top_k)
        except Exception as e:
            logger.error("Query failed: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Query failed: {e}"
            ) from e
        
        return QueryResponse(
            results=result.get("results", []),
            needs_retrieval=result.get("needs_retrieval", True)
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(_request: Any, e: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", e)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    return app
