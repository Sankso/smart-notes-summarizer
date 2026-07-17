#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main orchestrator for the Smart Notes Summarizer.

Implements an agentic multi-agent pipeline that processes documents chunk-by-chunk:

    PDF Upload → Parse & Extract → Sliding-Window Chunking → Keywords/chunk
    → Gemini routes/chunk → FLAN-T5 or Gemini/chunk → Merge & Dedup → Output

Each chunk is independently analyzed, keyword-extracted, routed, and summarized.
The final output merges all chunk summaries into a coherent semantic summary.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List

from agent.planner import PlannerAgent
from agent.executor import Executor
from agent.pdf_processor import PDFProcessor
from agent.keyword_extractor import KeywordExtractor
from agent.brain import GeminiSupervisor

logger = logging.getLogger(__name__)


class SmartSummarizerAgent:
    """
    Central orchestrator for the per-chunk multi-agent summarization pipeline.
    
    Pipeline flow:
    1. PDF/text ingestion → PDFProcessor parses unstructured documents
    2. Sliding-window chunking → Split text into overlapping chunks
    3. Per-chunk keyword extraction → YAKE + RAKE + TF-IDF/spaCy on each chunk
    4. Per-chunk routing → GeminiSupervisor decides: FLAN-T5 vs Gemini API
    5. Per-chunk summarization → Routed agent generates chunk summary
    6. Merge & deduplicate → Combine chunk summaries into final output
    """
    
    # Sliding-window parameters
    WINDOW_SIZE = 2000       # chars per chunk (~500 tokens for FLAN-T5)
    STRIDE = 1900            # step size (overlap = WINDOW_SIZE - STRIDE = 100 chars)
    MIN_CHUNK_LENGTH = 100   # skip chunks shorter than this
    
    def __init__(self,
                model_name: str = "google/flan-t5-small",
                lora_weights_dir: Optional[str] = None):
        """
        Initialize the Smart Summarizer Agent and all sub-agents.
        
        Args:
            model_name: Base model name for the executor
            lora_weights_dir: Directory containing LoRA adapter weights
        """
        logger.info("Initializing Smart Notes Summarizer Agent")
        
        # Initialize sub-agents
        self.planner = PlannerAgent()
        self.pdf_processor = PDFProcessor(ocr_enabled=True)
        self.keyword_extractor = KeywordExtractor()
        self.brain = GeminiSupervisor()
        
        # Find model weights directory if not specified
        if lora_weights_dir is None:
            default_weights_dir = os.path.join(os.path.dirname(__file__), "../models/lora_weights")
            if os.path.exists(default_weights_dir):
                lora_weights_dir = default_weights_dir
                logger.info(f"Using fine-tuned model at: {default_weights_dir}")
            else:
                potential_paths = ["./models/lora_weights", "./finetuning/lora_weights"]
                for path in potential_paths:
                    if os.path.exists(path):
                        lora_weights_dir = path
                        logger.info(f"Found LoRA weights at: {path}")
                        break
        
        # Initialize executor with model
        self.executor = Executor(
            model_name=model_name,
            model_path=lora_weights_dir
        )
        
        logger.info("Smart Notes Summarizer Agent initialized successfully")
    
    # ──────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────
    
    def summarize_pdf(self, pdf_path: str, summary_length: str = "normal",
                     extract_keywords: bool = True) -> Dict[str, Any]:
        """
        Full pipeline: PDF → Parse → Chunk → Keywords → Route → Summarize → Merge.
        
        Args:
            pdf_path: Path to the PDF file
            summary_length: 'short', 'normal', or 'long'
            extract_keywords: Whether to include keywords in output
            
        Returns:
            Dictionary with 'summary', 'keywords', 'chunk_details', and 'stats'
        """
        logger.info(f"Processing PDF: {pdf_path}")
        start_time = time.time()
        
        # Step 1: PDF Parsing & Text Extraction
        text = self.pdf_processor.extract_text(pdf_path)
        
        if not text.strip():
            logger.warning(f"No text extracted from PDF: {pdf_path}")
            return {"summary": "Error: No text could be extracted from the provided PDF."}
        
        # Run the per-chunk pipeline
        result = self.summarize_text(text, summary_length, extract_keywords)
        
        process_time = time.time() - start_time
        result["stats"]["source"] = os.path.basename(pdf_path)
        result["stats"]["processing_time_seconds"] = round(process_time, 2)
        
        logger.info(f"PDF processed in {process_time:.2f} seconds")
        return result
    
    def summarize_text(self, text: str, summary_length: str = "normal",
                      extract_keywords: bool = True) -> Dict[str, Any]:
        """
        Per-chunk multi-agent pipeline for text summarization.
        
        Pipeline:
        1. Sliding-window chunking
        2. For each chunk:
           a. Extract keywords (YAKE + RAKE + TF-IDF/spaCy)
           b. Gemini Supervisor routes chunk (simple → FLAN-T5, complex → Gemini)
           c. Routed agent summarizes the chunk
        3. Merge & deduplicate all chunk summaries
        
        Args:
            text: Raw text content to summarize
            summary_length: 'short', 'normal', or 'long'
            extract_keywords: Whether to include keywords in output
            
        Returns:
            Dictionary with 'summary', 'keywords', 'chunk_details', and 'stats'
        """
        logger.info(f"Starting per-chunk pipeline ({len(text)} chars)")
        start_time = time.time()
        
        # ── Step 1: Sliding-Window Chunking ──
        chunks = self._create_chunks(text)
        logger.info(f"Split text into {len(chunks)} chunks "
                    f"(window={self.WINDOW_SIZE}, stride={self.STRIDE})")
        
        # ── Steps 2-4: Per-chunk processing ──
        chunk_summaries = []
        chunk_details = []
        all_keywords = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"-- Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars) --")
            
            # Step 2: Keyword extraction per chunk
            chunk_keywords = []
            if extract_keywords:
                try:
                    chunk_keywords = self.keyword_extractor.extract_keywords(
                        chunk, method="combined"
                    )
                    logger.info(f"  Chunk {i+1}: extracted {len(chunk_keywords)} keywords")
                except Exception as e:
                    logger.warning(f"  Chunk {i+1}: keyword extraction failed: {e}")
            
            all_keywords.extend(chunk_keywords)
            
            # Step 2.5: Planner analyzes chunk complexity
            complexity_info = self.planner.analyze(chunk)
            logger.info(f"  Chunk {i+1}: complexity={complexity_info['complexity_score']:.2f}, "
                        f"action={complexity_info['action']}")
            
            # Step 3: Gemini Supervisor routes this chunk (informed by complexity analysis)
            decision = self.brain.decide_strategy(
                chunk_text=chunk,
                chunk_keywords=chunk_keywords,
                complexity_info=complexity_info,
                chunk_index=i,
                total_chunks=len(chunks)
            )
            action = decision.get("action", "summarize_local")
            chunk_length = decision.get("summary_length", summary_length)
            logger.info(f"  Chunk {i+1}: routed -> {action}")
            
            # Step 4: Summarize via the routed agent
            if action in ["summarize_gemini", "rewrite_gemini"]:
                chunk_summary = self.brain.process_with_gemini(chunk, action, chunk_length)
            else:
                # Default: local FLAN-T5 + LoRA
                chunk_summary = self.executor.generate_summary(chunk, length=chunk_length)
            
            chunk_summaries.append(chunk_summary)
            chunk_details.append({
                "chunk_index": i,
                "chunk_chars": len(chunk),
                "routing": action,
                "reason": decision.get("reason", ""),
                "keywords": [kw["keyword"] for kw in chunk_keywords[:5]],
                "summary_preview": chunk_summary[:100] + "..." if len(chunk_summary) > 100 else chunk_summary
            })
        
        # ── Step 5: Merge & Deduplicate ──
        logger.info(f"Merging {len(chunk_summaries)} chunk summaries...")
        merged_summary = self._merge_and_deduplicate(chunk_summaries)
        
        # ── Step 6: Summarize the Merged Summary ──
        if len(chunk_summaries) > 1:
            logger.info("Generating final comprehensive summary from merged chunks...")
            # Route the merged summary through Gemini for a final compression
            final_summary = self.brain.process_with_gemini(
                text=merged_summary, 
                action="summarize_gemini", 
                length=summary_length
            )
            # If Gemini fails, fallback to local model
            if final_summary.startswith("Error"):
                logger.warning("Gemini final summary failed, falling back to local model.")
                final_summary = self.executor.generate_summary(merged_summary, length=summary_length)
        else:
            # Only one chunk, so no need to double-summarize
            final_summary = merged_summary
            
        # ── Step 7: Final Cleanup ──
        # Run a final normalization pass to guarantee no model-generated citations slip through
        final_summary = self.executor.normalize_text(final_summary)
        
        # Deduplicate keywords across chunks
        unique_keywords = self._deduplicate_keywords(all_keywords)
        
        process_time = time.time() - start_time
        
        # Build result
        result = {
            "summary": final_summary,
            "chunk_details": chunk_details,
            "stats": {
                "input_chars": len(text),
                "num_chunks": len(chunks),
                "chunks_local": sum(1 for d in chunk_details if d["routing"] == "summarize_local"),
                "chunks_gemini": sum(1 for d in chunk_details if d["routing"] != "summarize_local"),
                "processing_time_seconds": round(process_time, 2)
            }
        }
        
        if extract_keywords:
            result["keywords"] = unique_keywords
        
        logger.info(f"Pipeline complete: {len(chunks)} chunks -> {len(final_summary)} char summary "
                    f"in {process_time:.2f}s")
        return result
    
    # ──────────────────────────────────────────────
    #  Pipeline Steps
    # ──────────────────────────────────────────────
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks using a sliding-window strategy.
        
        Optimizes context window performance by ensuring:
        - Each chunk fits within FLAN-T5's 512-token context (~2000 chars)
        - Overlap between windows prevents data loss at boundaries
        - Short texts (single-chunk) pass through without splitting
        
        Args:
            text: Full document text
            
        Returns:
            List of text chunks
        """
        # Normalize text first
        text = self.executor.normalize_text(text)
        
        if not text:
            return []
        
        # If text fits in a single window, no need to chunk
        if len(text) <= self.WINDOW_SIZE:
            return [text]
        
        chunks = []
        for i in range(0, len(text), self.STRIDE):
            chunk = text[i:i + self.WINDOW_SIZE]
            if len(chunk.strip()) >= self.MIN_CHUNK_LENGTH:
                chunks.append(chunk)
            if i + self.WINDOW_SIZE >= len(text):
                break
        
        return chunks
    
    def _merge_and_deduplicate(self, chunk_summaries: List[str]) -> str:
        """
        Merge multiple chunk summaries into a single coherent summary
        and remove duplicate/near-duplicate sentences across chunks.
        
        Args:
            chunk_summaries: List of summaries, one per chunk
            
        Returns:
            Merged and deduplicated final summary
        """
        if not chunk_summaries:
            return "No content to summarize."
        
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]
        
        # Concatenate all chunk summaries
        merged = " ".join(s.strip() for s in chunk_summaries if s.strip())
        
        # Use the executor's post-processing to deduplicate across chunks
        final_summary = self.executor.post_process_summary(merged)
        
        return final_summary
    
    def _deduplicate_keywords(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate keywords collected from multiple chunks.
        
        When the same keyword appears across chunks, keeps the one with
        the best score. Returns the top-N unique keywords.
        
        Args:
            keywords: Combined keyword list from all chunks
            
        Returns:
            Deduplicated and ranked keyword list
        """
        if not keywords:
            return []
        
        best_keywords = {}
        for kw in keywords:
            keyword = kw.get("keyword", "").lower().strip()
            if not keyword:
                continue
                
            score = kw.get("score", 0)
            method = kw.get("method", "")
            
            # For YAKE, lower score is better — invert for comparison
            normalized_score = (1 - score) if method == "YAKE" else score
            
            if keyword not in best_keywords or normalized_score > best_keywords[keyword]["_normalized_score"]:
                best_keywords[keyword] = {
                    "keyword": kw["keyword"],
                    "score": score,
                    "method": method,
                    "_normalized_score": normalized_score
                }
        
        # Sort by normalized score and return top 15
        ranked = sorted(best_keywords.values(), key=lambda x: x["_normalized_score"], reverse=True)
        
        # Remove internal scoring field
        for kw in ranked:
            del kw["_normalized_score"]
        
        return ranked[:15]