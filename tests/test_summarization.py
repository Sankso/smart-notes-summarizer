#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the per-chunk Smart Notes Summarizer pipeline.

Covers:
- Sliding-window chunking
- Per-chunk keyword extraction
- Per-chunk Gemini routing (fallback mode)
- Per-chunk FLAN-T5 summarization
- Merge & deduplication
- Full integration pipeline
"""

import pytest
import logging

logging.basicConfig(level=logging.INFO)


# ---------- Sliding-Window Chunking ----------

class TestChunking:
    """Test the sliding-window chunking strategy."""
    
    def test_short_text_single_chunk(self):
        """Short text should produce a single chunk (no splitting)."""
        from agent.agent import SmartSummarizerAgent
        agent = SmartSummarizerAgent(model_name="google/flan-t5-small")
        text = "This is a short sentence that fits in one window."
        chunks = agent._create_chunks(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        """Long text should be split into multiple overlapping chunks."""
        from agent.agent import SmartSummarizerAgent
        agent = SmartSummarizerAgent(model_name="google/flan-t5-small")
        # Create text longer than WINDOW_SIZE (2000 chars)
        long_text = "Natural language processing is a field of study. " * 100
        chunks = agent._create_chunks(long_text)
        assert len(chunks) > 1
        # Each chunk should be at most WINDOW_SIZE
        for chunk in chunks:
            assert len(chunk) <= agent.WINDOW_SIZE

    def test_chunks_have_overlap(self):
        """Adjacent chunks should overlap to prevent data loss."""
        from agent.agent import SmartSummarizerAgent
        agent = SmartSummarizerAgent(model_name="google/flan-t5-small")
        long_text = "Word " * 1000  # ~5000 chars
        chunks = agent._create_chunks(long_text)
        if len(chunks) >= 2:
            overlap = agent.WINDOW_SIZE - agent.STRIDE
            # The end of chunk[0] should overlap with the start of chunk[1]
            assert overlap > 0

    def test_empty_text_no_chunks(self):
        """Empty text should produce no chunks."""
        from agent.agent import SmartSummarizerAgent
        agent = SmartSummarizerAgent(model_name="google/flan-t5-small")
        chunks = agent._create_chunks("")
        assert len(chunks) == 0


# ---------- Executor (Single-Chunk) ----------

class TestExecutor:
    """Test FLAN-T5 + LoRA executor on single chunks."""
    
    def test_executor_initialization(self):
        from agent.executor import Executor
        executor = Executor(model_name="google/flan-t5-small")
        assert executor.device in ("cuda", "cpu")
        assert executor.tokenizer is not None

    def test_generate_summary_single_chunk(self):
        from agent.executor import Executor
        executor = Executor(model_name="google/flan-t5-small")
        text = (
            "Machine learning is a subset of artificial intelligence that provides "
            "systems the ability to automatically learn and improve from experience "
            "without being explicitly programmed."
        )
        summary = executor.generate_summary(text, length="short")
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_generate_summary_empty(self):
        from agent.executor import Executor
        executor = Executor(model_name="google/flan-t5-small")
        result = executor.generate_summary("")
        assert result == "No text content to summarize."

    def test_post_process_removes_duplicates(self):
        from agent.executor import Executor
        executor = Executor(model_name="google/flan-t5-small")
        text_with_dups = "This is a test. This is a test. Another sentence here."
        result = executor.post_process_summary(text_with_dups)
        assert result.count("This is a test") == 1


# ---------- Keyword Extractor (Per-Chunk) ----------

class TestKeywordExtractor:
    """Test multi-algorithmic keyword extraction on individual chunks."""
    
    def test_combined_extraction(self):
        from agent.keyword_extractor import KeywordExtractor
        extractor = KeywordExtractor()
        text = (
            "Deep learning neural networks have revolutionized computer vision "
            "and natural language processing. Convolutional neural networks are "
            "used for image classification while transformer models handle text."
        )
        keywords = extractor.extract_keywords(text, method="combined")
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        for kw in keywords:
            assert "keyword" in kw
            assert "score" in kw
            assert "method" in kw


# ---------- Gemini Supervisor (Per-Chunk Routing) ----------

class TestGeminiSupervisor:
    """Test per-chunk routing decisions (fallback mode without API key)."""
    
    def test_fallback_routing_per_chunk(self):
        from agent.brain import GeminiSupervisor
        supervisor = GeminiSupervisor(api_key=None)
        decision = supervisor.decide_strategy(
            chunk_text="Some academic text about machine learning.",
            chunk_keywords=[{"keyword": "machine learning", "score": 0.5, "method": "YAKE"}],
            chunk_index=0,
            total_chunks=3
        )
        assert decision["action"] == "summarize_local"
        assert "Fallback" in decision["reason"]

    def test_fallback_returns_all_required_fields(self):
        from agent.brain import GeminiSupervisor
        supervisor = GeminiSupervisor(api_key=None)
        decision = supervisor.decide_strategy(chunk_text="Test text.")
        assert "action" in decision
        assert "summary_length" in decision
        assert "reason" in decision


# ---------- Merge & Deduplication ----------

class TestMerge:
    """Test merging and deduplication of chunk summaries."""
    
    def test_merge_single_summary(self):
        from agent.agent import SmartSummarizerAgent
        agent = SmartSummarizerAgent(model_name="google/flan-t5-small")
        result = agent._merge_and_deduplicate(["This is a single summary."])
        assert result == "This is a single summary."

    def test_merge_deduplicates_across_chunks(self):
        from agent.agent import SmartSummarizerAgent
        agent = SmartSummarizerAgent(model_name="google/flan-t5-small")
        summaries = [
            "Machine learning is a field of AI.",
            "Machine learning is a field of AI. It uses neural networks."
        ]
        result = agent._merge_and_deduplicate(summaries)
        # The duplicate sentence should appear only once
        assert result.count("Machine learning is a field of AI") == 1

    def test_merge_empty_list(self):
        from agent.agent import SmartSummarizerAgent
        agent = SmartSummarizerAgent(model_name="google/flan-t5-small")
        result = agent._merge_and_deduplicate([])
        assert result == "No content to summarize."

    def test_deduplicate_keywords(self):
        from agent.agent import SmartSummarizerAgent
        agent = SmartSummarizerAgent(model_name="google/flan-t5-small")
        keywords = [
            {"keyword": "machine learning", "score": 0.3, "method": "YAKE"},
            {"keyword": "machine learning", "score": 0.1, "method": "YAKE"},  # better YAKE (lower)
            {"keyword": "neural networks", "score": 0.8, "method": "TF-IDF"},
        ]
        result = agent._deduplicate_keywords(keywords)
        # Should have 2 unique keywords
        kw_names = [kw["keyword"].lower() for kw in result]
        assert len(set(kw_names)) == 2


# ---------- Full Integration ----------

class TestIntegration:
    """End-to-end integration test for the per-chunk pipeline."""
    
    def test_full_pipeline_short_text(self):
        """Short text: single chunk, full pipeline."""
        from agent.agent import SmartSummarizerAgent
        agent = SmartSummarizerAgent(model_name="google/flan-t5-small")
        
        text = (
            "Artificial intelligence has made significant strides in recent years, "
            "particularly in areas such as natural language processing and computer "
            "vision. Deep learning models have achieved state-of-the-art performance "
            "on a wide range of benchmarks."
        )
        
        result = agent.summarize_text(text, summary_length="short", extract_keywords=True)
        
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0
        assert "keywords" in result
        assert "chunk_details" in result
        assert "stats" in result
        assert result["stats"]["num_chunks"] == 1

    def test_full_pipeline_long_text(self):
        """Long text: multiple chunks, each independently routed."""
        from agent.agent import SmartSummarizerAgent
        agent = SmartSummarizerAgent(model_name="google/flan-t5-small")
        
        long_text = (
            "Natural language processing involves the interaction between computers "
            "and humans through natural language. The field includes tasks such as "
            "text classification, named entity recognition, and machine translation. "
        ) * 60  # ~200+ words repeated → >2000 chars
        
        result = agent.summarize_text(long_text, summary_length="short", extract_keywords=True)
        
        assert "summary" in result
        assert len(result["summary"]) > 0
        assert result["stats"]["num_chunks"] > 1
        # Verify chunk details are tracked
        assert len(result["chunk_details"]) == result["stats"]["num_chunks"]
        # Each chunk detail should have routing info
        for detail in result["chunk_details"]:
            assert "routing" in detail
            assert "keywords" in detail
