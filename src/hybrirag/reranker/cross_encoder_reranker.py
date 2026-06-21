"""Cross-encoder reranker using a Hugging Face sequence-classification model."""

import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

class RerankerError(Exception):
    """Raised when reranking operations fail."""

class CrossEncoderReranker:
    """Re-rank candidate documents with a cross-encoder model."""

    def __init__(
        self,
        model_path: str = "BAAI/bge-reranker-base",
        device: str = "cuda"
    ) -> None:
        self.model_path = model_path
        self.device = self._reslove_device(device)
        self._model = None
        self._tokenizer = None

        logger.info(
            "CrossEncoderReranker initialised: model=%s, device=%s",
            self.model_path,
            self.device,
        )

    def _load_model(self) -> None:
        """Load the model and tokenizer on first use."""

        if self._model is not None:
            return
        
        try:
            logger.info("Loading reranker model '%s' on %s ...", self.model_path, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            )
            self._model.to(self.device)
            self._model.eval()
            logger.info("Reranker model loaded.")
        except Exception as e:
            raise RerankerError(
                f"Failed to load reranker model '{self.model_path}': {e}"
            ) from e
        
    def rerank(
        self,
        query: str,
        documents: list[str],
        doc_ids: list[int],
        top_k: int = 5
    ) -> list[dict]:
        """
        Score and rank *documents* against *query*.
        
        Args:
            query: The search query.
            documents: Candidate document texts.
            doc_ids: Integer IDs corresponding to each document.
            top_k: Number of top results to return.

        Returns:
            list[dict] Each dict has keys ``doc_id``, ``score``, ``text``
        """

        if not documents:
            return []
        
        if len(documents) != len(doc_ids):
            raise RecursionError("documents and doc_ids must have the same length")
        
        self._load_model()

        top_k = min(top_k, len(documents))
        scores = self._batch_predict(query, documents)

        # Pair up and sort.
        paired = list(zip(doc_ids, scores, documents))
        paired.sort(key=lambda x: x[1], reverse=True)

        results = [
            {"doc_id": did, "score": float(score), "text": text}
            for did, score, text in paired[:top_k]
        ]

        logger.debug("Reranked %d documents, returning top %d.", len(documents), top_k)
        return results
    
    def _batch_predict(self, query: str, documents: list[str]) -> list[float]:
        batch_size = 16
        all_scores: list[float] = []

        assert self._tokenizer is not None
        assert self._model is not None

        with torch.no_grad():
            for start in range(0, len(documents), batch_size):
                batch_docs = documents[start : start + batch_size]
                pairs = [[query, doc] for doc in batch_docs]

                try:
                    tokens = self._tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    tokens = {k: v.to(self.device) for k, v in tokens.items()}

                    outputs = self._model(**tokens)  

                    # For single-label classification the logits shape is (batch, 1).
                    logits = outputs.logits.squeeze(-1)
                    batch_scores = logits.cpu().tolist()

                    # Normalise to a list even for a single-element batch.
                    if isinstance(batch_scores, (int, float)):
                        batch_scores = [batch_scores]
                    
                    all_scores.extend(batch_scores)
                except Exception as e:
                    logger.warning(
                        "Reranker batch inference failed for batch starting at %d: %s",
                        start,
                        e,
                    )
                    all_scores.extend([0.0] * len(batch_docs))
        return all_scores

    @staticmethod
    def _reslove_device(device: str) -> str:
        if device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device
    