"""Dense (vector) retriever backed by a FAISS inner-product index."""

from pathlib import Path
import logging

import faiss
import numpy as np

logger = logging.getLogger(__name__)

class DenseRetrieverError(Exception):
    """Raised when dense retrieval operations fail."""

class DenseRetriever:
    """FAISS-based dense vector retriever using inner-product (cosine) search."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self._id_map: dict[int, int] = {}
        self._rev_id_map: dict[int, int] = {}
        self._next_idx: int = 0
        self._index = faiss.IndexFlatIP(self.dimension)
        logger.info("Created FAISS IndexFlatIP with dimension=%d", self.dimension)

    def add(self, embeddings: np.ndarray, ids: list[int]) -> None:
        """
        Add embedding vectors with their associated document IDs.

        Args:
            embeddings: Array of shape (n, dimension) with L2-normalised rows.
            ids: External integer IDs.
        """

        if len(embeddings) != len(ids):
            raise DenseRetrieverError(
                f"embeddings length ({len(embeddings)}) != ids length ({len(ids)})"
            )
        
        if embeddings.shape[1] != self.dimension:
            raise DenseRetrieverError(
                f"embedding dimension {embeddings.shape[1]} != "
                f"index dimension {self.dimension}"
            )
        
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        for ext_id in ids:
            int_idx = self._next_idx
            self._id_map[ext_id] = int_idx
            self._rev_id_map[int_idx] = ext_id
            self._next_idx += 1

        self._index.add(embeddings)
        logger.debug("Added %d vectors to FAISS index (total: %d).", len(ids), self.count)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[tuple[int, float]]:
        """
        Search the index for vectors closest to *query_embedding*.

        Args:
            query_embedding: Query vector
            top_k: Number of nearest neighbours to return.

        Returns:
            List of (external_id, score) sorted by descending score.
        """

        if self.count == 0:
            return []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

        scores, indices = self._index.search(query_embedding, min(top_k, self.count))

        results: list[tuple[int, float]] = []
        for score, int_idx in zip(scores[0], indices[0]):
            if int_idx < 0:
                continue
            ext_id = self._rev_id_map.get(int_idx)
            if ext_id is not None:
                results.append((ext_id, float(score)))

        return results
    
    def save(self, path: str | Path) -> None:
        """
        Persist the FAISS index and ID mappings

        Args:
            path: dir of saved files
        """

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "index.faiss"))

        ext_ids = np.array(
            list(self._id_map.keys()), dtype=np.int64
        )
        int_ids = np.array(
            list(self._id_map.values()), dtype=np.int64
        )
        np.savez(str(path / "id_map.npz"), ext_ids=ext_ids, int_ids=int_ids)
        logger.info("Dense retriever saved to %s", path)

    def load(self, path: str | Path) -> None:
        """
        Load a previously saved FAISS index and ID mappings.

        Args:
            path: dir of index and napping file.
        """

        path = Path(path)

        index_path = path / "index.faiss"
        map_path = path / "id_map.npz"

        if not index_path.exists():
            raise DenseRetrieverError(f"Index file not found: {index_path}")
        
        self._index = faiss.read_index(str(index_path))

        if map_path.exists():
            data = np.load(str(map_path))
            ext_ids = data["ext_ids"]
            int_ids = data["int_ids"]
            self._id_map = dict(zip(ext_ids.tolist(), int_ids.tolist()))
            self._rev_id_map = dict(zip(int_ids.tolist(), ext_ids.tolist()))
            self._next_idx = int(int_ids.max()) + 1 if len(int_ids) > 0 else 0
        else:
            logger.warning("ID map file not found at %s; ID mappings will be empty.", map_path)
            self._id_map = {}
            self._rev_id_map = {}
            self._next_idx = self._index.ntotal

        logger.info("Dense retriever loaded from %s (%d vectors)", path, self.count)

    @property
    def count(self) -> int:
        """Number of vectors currently in the index."""

        return self._index.ntotal if self._index is not None else 0
