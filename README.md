# HybriRAG

一个双路召回系统（BM25 + 向量），然后用 Cross-Encoder 对召回结果进行二次打分。同时，引入自适应的语义分块策略，处理长文本的上下文截断问题。

## Architecture

```
Query → Query Router → [Dense Retriever + Sparse Retriever] → RRF Fusion → Cross-Encoder Reranker → Top-K Results
                           │                │
                      FAISS Index      C++ BM25 Engine
                      (BGE Embedding)    (pybind11)
```

## Features

- **Hybrid Retrieval** — 通过 RRF 结合稠密（FAISS + BGE 语义嵌入）和稀疏（BM25）检索，兼顾语义相似性与关键词精确匹配
- **C++ Accelerated BM25** — C++17 高性能BM25引擎通过 pybind11 绑定
- **Cross-Encoder Reranking** — BGE-reranker-base 交叉编码器在 query 和文档之间应用完整的注意力机制，实现精确的相关性评分
- **Self-RAG Routing** — query 路由会跳过问候语和简短查询，并检查候选结果的相关性，以避免出现低质量的搜索结果
- **Semantic Chunking** — 通过 spacy 和滑动窗口重叠方法，实现语句分块
- **REST API** — 异步端点的 FastAPI 服务器，用于文档导入、查询和健康检查

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Jake-Wey/HybriRAG.git
cd HybriRAG

# Install spaCy language model for chunking
python -m spacy download en_core_web_sm

Build C++ BM25 library
```

### Python SDK

```python
from hybrirag.engine import HybriRAGEngine

# Create the engine
engine = HybriRAGEngine(
    embedding_model_name="BAAI/bge-small-en-v1.5",
    rerank_model_name="BAAI/bge-reranker-base",
)

# Add documents
documents = [
    "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.",
    "BM25 is a probabilistic ranking function based on term frequency and inverse document frequency.",
    "Dense retrieval uses neural embeddings to find semantically similar documents in vector space.",
]
engine.add_documents(documents)

# Query
results = engine.retrieve("How does BM25 rank documents?", top_k=3)
for r in results:
    print(f"[score={r['score']:.4f}] {r['text'][:100]}...")
```

## Project Structure

```
HybriRAG/
├── src/
│   ├── hybrirag/                 # Main package
│   │   ├── engine.py             # Pipeline orchestrator
│   │   ├── chunker/              # Semantic text chunking
│   │   ├── embedder/             # BGE sentence embeddings
│   │   ├── retriever/            # Dense (FAISS) + Sparse (BM25) + Hybrid (RRF)
│   │   ├── reranker/             # Cross-encoder reranking
│   │   ├── router/               # Self-RAG query routing
│   │   └── api/                  # FastAPI REST server
│   └── cpp_bm25/                 # C++17 BM25 engine (pybind11)
│       ├── include/              # Headers: inverted_index, bm25_engine, tokenizer
│       └── src/                  # Implementation + Python bindings
├── tests/                        # Unit tests
└── examples/                     # Usage examples
    ├── basic_usage.py            # Python SDK demo
    └── api_demo.py               # REST API demo
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_model_name` | `BAAI/bge-small-en-v1.5` | Bi-encoder model for dense retrieval |
| `rerank_model_name` | `BAAI/bge-reranker-base` | Cross-encoder model for reranking |
| `device` | `"cuda"` | Compute device: `"cuda"` (auto-fallback to CPU) or `"cpu"` |
| `chunk_min_size` | `50` | Minimum chunk size in characters |
| `chunk_max_size` | `500` | Maximum chunk size in characters |
| `chunk_overlap` | `1` | Overlap sentences between chunks |

## Requirements

- Python >= 3.11
- PyTorch >= 2.5
- C++17 compiler
- CUDA-capable GPU (可选, 推荐用于模型推理)

## References

- **RAG** — Lewis, P., Perez, E., Piktus, A., et al. (2020). 
- **BM25** — Robertson, S. E., & Zaragoza, H. (2009). 
- **Reciprocal Rank Fusion** — Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009).

## License

MIT
