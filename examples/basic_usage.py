"""Basic usage example for HybriRAG."""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))
sys.path.append(src_root)

from hybrirag.engine import HybriRAGEngine

def main() -> None:
    engine = HybriRAGEngine(
        embedding_model=r"E:\Models\Embedding\BAAI-bge-small-en-v1.5",
        rerank_model=r"E:\Models\Reranker\BAAI-bge-reranker-base"
    )

    documents = [
        "Retrieval-Augmented Generation (RAG) is a technique that combines "
        "information retrieval with text generation. It allows language models "
        "to access external knowledge at inference time, which significantly "
        "reduces hallucination and improves factual accuracy.",

        "BM25 is a probabilistic ranking function used in information retrieval. "
        "It is based on the term frequency and inverse document frequency of "
        "query terms in each document. BM25 excels at exact keyword matching "
        "but does not understand semantic similarity.",

        "Dense retrieval encodes queries and documents into continuous vector "
        "embeddings using neural models such as sentence transformers. Similarity "
        "search is then performed in vector space using libraries like FAISS. "
        "Dense retrieval captures semantic meaning but may miss exact term matches.",
        
        "Hybrid retrieval combines the strengths of both sparse and dense methods. "
        "Dense retrieval captures semantic similarity while sparse retrieval excels "
        "at exact keyword matching. The two result lists are fused using algorithms "
        "like Reciprocal Rank Fusion (RRF) to produce a unified ranking.",
   
        "Cross-encoder reranking applies a more computationally expensive model "
        "that jointly encodes the query and each candidate document. Unlike "
        "bi-encoders, cross-encoders allow full attention between query and "
        "document tokens, producing highly accurate relevance scores at the cost "
        "of slower inference."
    ]

    print("Adding documents to the engine...")
    engine.add_documents(documents)
    print(f"Indexed {len(documents)} documents.\n")

    queries = [
        "How does BM25 rank documents?",
        "What is the advantage of hybrid retrieval?",
        "Why use a cross-encoder for reranking?",
    ]

    for query in queries:
        print(f"Query: {query}")
        results = engine.retrieve(query, top_k=3)
        for i, result in enumerate(results, 1):
            doc_id = result.get("id", "N/A")
            score = result.get("score", 0.0)
            text = result.get("text", "")
            print(f"  [{i}] id={doc_id}  score={score:.4f}")
            print(f"      {text[:120]}...")
        print()

if __name__ == "__main__":
    main()
