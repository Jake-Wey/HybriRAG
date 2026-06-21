"""Text embedding using sentence-transformers models."""

import logging

import torch
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Raised when text embedding fails."""

class Embedder:
    """Encode texts into dense vectors using a sentence-transformers model."""

    def __init__(
        self,
        model_path: str = "BAAI/bge-small-en-v1.5",
        device: str = "cuda"
    ) -> None:
        self.model_path = model_path
        self.device = self._reslove_device(device)
        self._model = None
        self._dimension: int | None = None
        self._supports_instruction: bool = "bge" in model_path.lower()

        logger.info(
            "Embedder initialised: model=%s, device=%s", self.model_path, self.device
        )

    def _load_model(self) -> None:
        if self._model is not None:
            return
        
        try:
            logger.info("Loading embedding model '%s' on %s ...", 
                        self.model_path, self.device)
            self._model = SentenceTransformer(
                self.model_path, device=self.device
            )
            self._dimension = self._model.get_embedding_dimension()
            logger.info("Model loaded. Embedding dimension: %d", self._dimension)
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load embedding model '{self.model_path}': {e}"
            ) from e
        
    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""

        self._load_model()
        assert self._dimension is not None
        return self._dimension
    
    def encode(
        self, texts: list[str], batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode a list of texts into dense vectors.

        Args:
            texts: Documents or passages to encode.
            batch_size: Number of texts processed per forward pass.

        Returns:
            L2-normalised rows
        """

        if not texts:
            return np.ndarray(shape=(0, 0), dtype=np.float32)
        
        self._load_model()
        assert self._model is not None

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as e:
            raise EmbeddingError(f"Encoding failed: {e}") from e

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query

        Args:
            query: The search query string.

        Returns:
            Normalised query vector of shape ``(dimension,)``.
        """
        self._load_model()

        text = query
        if self._supports_instruction:
            text = f"Represent this sentence for searching similar passages: {query}"

        result = self.encode([text], batch_size=1)
        return result[0]    

    @staticmethod
    def _reslove_device(device: str) -> str:
        if device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device
        