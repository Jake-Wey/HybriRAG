"""API demo for HybriRAG."""

import threading
import time

import uvicorn
import httpx

from hybrirag.api.server import create_app

BASE_URL = "http://127.0.0.1:8900"

def run_server() -> None:
    """
    Start the FastAPI server in a background thread.
    """

    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8900, log_level="warning")

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Start the server in a background thread
    # ------------------------------------------------------------------

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("Starting HybriRAG API server...")
    time.sleep(3)

    client = httpx.Client(base_url=BASE_URL, timeout=30.0)

    # ------------------------------------------------------------------
    # 2. Health check
    # ------------------------------------------------------------------

    resp = client.get("/health")
    print(f"Health check: {resp.status_code} -> {resp.json()}\n")

    # ------------------------------------------------------------------
    # 3. Add documents
    # ------------------------------------------------------------------
    documents = [
        {
            "id": "doc1",
            "text": (
                "Retrieval-Augmented Generation (RAG) is a technique that combines "
                "information retrieval with text generation to reduce hallucination."
            ),
        },
        {
            "id": "doc2",
            "text": (
                "BM25 is a probabilistic ranking function based on term frequency "
                "and inverse document frequency, widely used in keyword search."
            ),
        },
        {
            "id": "doc3",
            "text": (
                "Dense retrieval uses neural embeddings to find semantically similar "
                "documents in vector space using approximate nearest neighbor search."
            ),
        },
    ]

    print("Adding documents via API...")
    resp = client.post("/documents", json={"documents": documents})
    print(f"Add documents: {resp.status_code} -> {resp.json()}\n")

    # ------------------------------------------------------------------
    # 4. Query the engine
    # ------------------------------------------------------------------
    queries = [
        "How does RAG reduce hallucination?",
        "What is BM25 used for?",
    ]

    for query in queries:
        print(f"Query: {query}")
        resp = client.post("/query", json={"query": query, "top_k": 3})
        if resp.status_code == 200:
            data = resp.json()
            for i, result in enumerate(data.get("results", []), 1):
                doc_id = result.get("id", "N/A")
                score = result.get("score", 0.0)
                text = result.get("text", "")
                print(f"  [{i}] id={doc_id}  score={score:.4f}")
                print(f"      {text[:120]}...")
        else:
            print(f"  Error: {resp.status_code} {resp.text}")
        print()

    print("Demo complete. Server is running in background (daemon thread).")
    print("The server will shut down when this script exits.")

if __name__ == "__main__":
    main()