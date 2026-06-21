"""Query router for deciding whether retrieval is necessary."""

import logging
import re

logger = logging.getLogger(__name__)

_GREETING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(hi|hello|hey|greetings|howdy|yo|sup)\b", re.IGNORECASE),
    re.compile(r"^(good\s*(morning|afternoon|evening|day|night))\b", re.IGNORECASE),
    re.compile(r"^(what'?s\s+up|how\s+are\s+you)\b", re.IGNORECASE),
    re.compile(r"^(thanks?|thank\s+you|bye|goodbye)\b", re.IGNORECASE),
]

_MIN_QUERY_TOKENS = 4

class RouterError(Exception):
    """Raised when query routing operations fail."""

class QueryRouter:
    """Decide whether a query needs retrieval and whether results are relevant."""

    def __init__(self, relevance_threshold: float = 0.3) -> None:
        if not 0.0 <= relevance_threshold <= 1.0:
            raise RouterError("relevance_threshold must be between 0.0 and 1.0")
        self.relevance_threshold = relevance_threshold
        logger.info(
            "QueryRouter initialised (relevance_threshold=%.2f)", self.relevance_threshold
        )

    def needs_retrieval(self, query: str) -> bool:
        """
        Return ``True`` if the query warrants document retrieval.

        Args:
            query: The user's raw query string.

        Returns:
            True if retrieval should proceed, False otherwise.
        """

        stripped = query.strip()
        if not stripped:
            logger.debug("Empty query -- skipping retrieval.")
            return False
        
        # Check greeting patterns.
        for pattern in _GREETING_PATTERNS:
            if pattern.match(stripped):
                logger.debug("Query matches greeting pattern -- skipping retrieval.")
                return False
        
        # Check token count.
        token_count = len(stripped.split())
        if token_count < _MIN_QUERY_TOKENS:
            logger.debug(
                "Query too short (%d tokens < %d) -- skipping retrieval.",
                token_count,
                _MIN_QUERY_TOKENS
            )
            return False
        
        return True
    
    def check_relevance(self, scores: list[float]) -> bool:
        """
        if at least one score exceeds the threshold.
        
        Args:
            scores: List of relevance scores.

        Returns:
            True if the results are considered relevant.
        """

        if not scores:
            return False
        
        max_score = max(scores)
        is_relevant = max_score >= self.relevance_threshold

        if not is_relevant:
            logger.debug(
                "Max score %.4f below threshold %.2f -- results not relevant.",
                max_score,
                self.relevance_threshold
            )

        return is_relevant
    