"""Semantic chunker for splitting text into overlapping sentence-based chunks."""

import unicodedata
import re
import logging

import spacy

logger = logging.getLogger(__name__)

class SemanticChunker:

    def __init__(
        self,
        nlp_model: str = "en_core_web_sm",
        min_chunk_size: int = 50,
        max_chunk_size: int = 500,
        overlap_sentences: int = 1
    ) -> None:
        self.nlp_model = nlp_model
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
        
        self._nlp = spacy.load(self.nlp_model)
        logger.info("Loaded spaCy model '%s' for sentence splitting.", self.nlp_model)

    def _split_sentences(self, text: str) -> list[str]:
        """
        Return a list of sentence strings detected in *text*.

        Args:
            text: str

        Returns:
            a list of sentence
        """

        doc = self._nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    def chunk(self, text: str) -> list[dict]:
        """
        Split *text* into sentence-aligned, overlapping chunks.
        
        Args:
            text: doc text.

        Returns:
            list[dict] Each dict has keys text, start, end,`chunk_id
        """

        if not text or not text.strip():
            return []
        
        cleaned = self._clean_text(text)
        sentences = self._split_sentences(cleaned)

        if not sentences:
            return []
        
        # Build (sentence, start, end) triples relative to *cleaned*.
        sentence_spans: list[tuple[str, int, int]] = []
        offset = 0
        for sent in sentences:
            idx = cleaned.find(sent, offset)
            if idx == -1:
                # Fallback: approximate position.
                idx = offset
            sentence_spans.append((sent, idx, idx + len(sent)))
            offset = idx + len(sent)
        
        raw_chunks = self._sliding_window(sentence_spans)
        return self._merge_small(raw_chunks)

    def _sliding_window(
        self, sentence_spans: list[tuple[str, int, int]]
    ) -> list[dict]:
        """
        Apply a sliding window over sentences to form chunks.

        Args:
            sentence_spans: raw sentence

        Returns:
            chunks
        """

        chunks: list[dict] = []
        chunk_id = 0
        i = 0
        n = len(sentence_spans)

        while i < n:
            chunk_sents: list[tuple[str, int, int]] = []
            char_count = 0

            j = i
            while j < n:
                sent_text, _, _ = sentence_spans[j]
                added_len = len(sent_text) + (1 if chunk_sents else 0) # space
                if char_count + added_len > self.max_chunk_size and chunk_sents:
                    break
                chunk_sents.append(sentence_spans[j])
                char_count += added_len
                j += 1
            
            text = " ".join(s[0] for s in chunk_sents)
            start = chunk_sents[0][1]
            end = chunk_sents[-1][2]

            chunks.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "chunk_id": chunk_id
                }
            )
            chunk_id += 1

            # Advance with overlap: step forward by (chunk_size - overlap).
            step = max(1, len(chunk_sents) - self.overlap_sentences)
            i += step
        
        return chunks

    def _merge_small(self, chunks: list[dict]) -> list[dict]:
        """
        Merge consecutive chunks that are below *min_chunk_size*.
        
        
        Args:
            chunks: consecutive chunks

        Returns:
            merged chunks
        """

        if not chunks:
            return []
        
        merged: list[dict] = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            if len(current["text"]) < self.min_chunk_size:
                current = {
                    "text": current["text"] + " " + next_chunk["text"],
                    "start": current["start"],
                    "end": current["end"],
                    "chunk_id": current["chunk_id"]
                }
            else:
                merged.append(current)
                current = next_chunk
        
        merged.append(current)

        # Re-index chunk_ids.
        for idx, chunk in enumerate(merged):
            chunk["chunk_id"] = idx

        return merged

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Normalise unicode and collapse excessive whitespace.
        
        Args:
            text: raw text
        """

        text = unicodedata.normalize("NFC", text)
        
        text = re.sub(r"[\t\r\f\v]+", " ", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        text = text.strip()
        return text
    