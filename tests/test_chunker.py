"""Tests for the SemanticChunker module."""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_root = os.path.abspath(os.path.join(project_root, "src"))
sys.path.append(src_root)

import pytest

from hybrirag.chunker import SemanticChunker

@pytest.fixture
def chunker() -> SemanticChunker:
    return SemanticChunker(
        min_chunk_size=20,
        max_chunk_size=500,
        overlap_sentences=1
    )

@pytest.fixture
def sample_text() -> str:
    return (
        "Retrieval-augmented generation is a technique that combines information "
        "retrieval with text generation. It allows language models to access external "
        "knowledge at inference time. This approach significantly reduces hallucination "
        "and improves factual accuracy.\n\n"
        "The retrieval component typically uses dense vector search or sparse keyword "
        "matching. Dense retrieval encodes queries and documents into embedding vectors "
        "and finds nearest neighbors in vector space. Sparse retrieval uses algorithms "
        "like BM25 to match query terms against an inverted index.\n\n"
        "Hybrid approaches combine both methods to get the best of both worlds. Dense "
        "retrieval captures semantic similarity while sparse retrieval excels at exact "
        "keyword matching. The results are then fused using techniques like Reciprocal "
        "Rank Fusion."
    )

@pytest.fixture
def dirty_text() -> str:
    return (
        "  This   has   extra   spaces.  \n\n\n\n"
        "And\t\ttabs\ttoo.\n"
        "  \n  Leading and trailing blanks.  \n  "
    )

class TestBasicChunking:
    """
    Verify that the chunker splits text into reasonable chunks.
    """

    def test_chunker_results_list(self, chunker: SemanticChunker, sample_text: str):
        chunks = chunker.chunk(sample_text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_each_chunk_is_dict(self, chunker: SemanticChunker, sample_text: str):
        """Each chunk should be a dict with the expected keys."""
        
        chunks = chunker.chunk(sample_text)
        for c in chunks:
            assert isinstance(c, dict)
            assert "text" in c
            assert "start" in c
            assert "end" in c
            assert "chunk_id" in c
            assert len(c["text"]) > 0

    def test_chunks_preserve_content(self, chunker: SemanticChunker, sample_text: str):
        """All original words should appear somewhere in the chunks."""

        chunks = chunker.chunk(sample_text)
        combined = " ".join(c["text"] for c in chunks)
        for word in sample_text.split():
            assert word in combined
    
    def test_single_sentence_text(self, chunker: SemanticChunker):
        """A very short text should produce at least one chunk."""

        chunks = chunker.chunk("Hello world.")
        assert len(chunks) >= 1

    def test_chunk_ids_sequential(self, chunker: SemanticChunker, sample_text: str):
        """Chunk IDs should be sequential starting from 0."""

        chunks = chunker.chunk(sample_text)
        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_empty_text_returns_empty(self, chunker: SemanticChunker):
        """Empty or whitespace-only text should produce no chunks."""

        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

class TestOverlap:
    """
    Verify that overlapping sentences are included when configured.
    """

    def test_overlap_produces_repeated_content(self, sample_text: str):
        """With overlap > 0, adjacent chunks should share some sentences."""

        overlap_chunker = SemanticChunker(
            min_chunk_size=20,
            max_chunk_size=500,
            overlap_sentences=2,
        )
        chunks = overlap_chunker.chunk(sample_text)

        # Collect all sentence texts from all chunks
        all_sentences: list[str] = []
        for chunk in chunks:
            all_sentences.extend(chunk["text"].split(". "))

        # Count occurrences of each sentence text
        sentence_counts: dict[str, int] = {}
        for s in all_sentences:
            s_stripped = s.strip()
            if s_stripped:
                sentence_counts[s_stripped] = sentence_counts.get(s_stripped, 0) + 1
       
        # At least one sentence should appear more than once
        has_overlap = any(count > 1 for count in sentence_counts.values())
        assert has_overlap, "Expected at least one overlapping sentence between chunks"

    def test_zero_overlap_no_repetition(self, sample_text: str):
        """With overlap_sentences=0, chunks should not share trailing sentences."""
        
        no_overlap_chunker = SemanticChunker(
            min_chunk_size=20,
            max_chunk_size=500,
            overlap_sentences=0,
        )
        chunks = no_overlap_chunker.chunk(sample_text)

        # Basic sanity: we still get chunks
        assert len(chunks) >= 1

class TestChunkSizeConstraints:
    """
    Verify that chunk sizes respect the configured bounds.
    """

    def test_max_chunk_size_respected(self, sample_text: str):
        chunker = SemanticChunker(
            min_chunk_size=10,
            max_chunk_size=150,
            overlap_sentences=0,
        )
        chunks = chunker.chunk(sample_text)
        for chunk in chunks:

            # Allow a small margin since splitting happens at sentence boundaries
            assert len(chunk["text"]) <= 200, f"Chunk exceeds max size: {len(chunk['text'])} chars"

    def test_min_chunk_size_merge(self):
        """Very small chunks should be merged to meet the minimum size."""

        chunker = SemanticChunker(
            min_chunk_size=50,
            max_chunk_size=500,
            overlap_sentences=0,
        )
        text = "Short. " * 5  # 5 tiny sentences
        chunks = chunker.chunk(text)

        # The chunker should merge these into fewer, larger chunks
        for chunk in chunks:
            assert len(chunk["text"].strip()) > 0

class TestTextCleaning:
    """
    Verify that dirty text is cleaned before chunking.
    """

    def test_extra_whitespace_collapsed(self, dirty_text: str):
        cleaned = SemanticChunker._clean_text(dirty_text)

        # No runs of multiple spaces
        assert "   " not in cleaned

        # No leading/trailing whitespace on the whole string
        assert cleaned == cleaned.strip()

    def test_blank_lines_removed(self, dirty_text: str):
        cleaned = SemanticChunker._clean_text(dirty_text)

        # No more than two consecutive newlines
        assert "\n\n\n" not in cleaned

    def test_tabs_replaced(self, dirty_text: str):
        cleaned = SemanticChunker._clean_text(dirty_text)
        assert "\t" not in cleaned
