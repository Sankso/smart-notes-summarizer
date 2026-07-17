#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemini Supervisor Brain for Smart Notes Summarizer.

Acts as the per-chunk decision agent in the pipeline. For each text chunk,
analyzes complexity and extracted keywords to dynamically route the chunk
to either the local fine-tuned FLAN-T5 model or the Gemini API.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

_GENAI_AVAILABLE = False
try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except (ImportError, Exception):
    genai = None

logger = logging.getLogger(__name__)


class GeminiSupervisor:
    """
    Per-chunk routing brain that decides whether each chunk should be
    processed by the local FLAN-T5 model or the Gemini API.
    
    Receives a text chunk and its extracted keywords, then uses Gemini
    to analyze complexity and make a routing decision. Falls back to
    local processing when the API is unavailable.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini Supervisor.
        
        Args:
            api_key: Gemini API key. If None, checks GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        if not _GENAI_AVAILABLE:
            logger.warning("google-generativeai package not available. Fallback mode.")
        elif not self.api_key:
            logger.warning("GEMINI_API_KEY not found. GeminiSupervisor will run in fallback mode.")
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-2.0-flash")
                logger.info("GeminiSupervisor initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
                self.model = None

    def decide_strategy(self, chunk_text: str,
                       chunk_keywords: Optional[List[Dict[str, Any]]] = None,
                       complexity_info: Optional[Dict[str, Any]] = None,
                       chunk_index: int = 0,
                       total_chunks: int = 1) -> Dict[str, Any]:
        """
        Analyze a single text chunk and decide the best processing strategy.
        
        Uses the chunk's content, extracted keywords, and complexity analysis
        to make an informed routing decision. Each chunk is independently evaluated.
        
        Args:
            chunk_text: The text content of this chunk
            chunk_keywords: Keywords extracted from this chunk
            complexity_info: Complexity analysis from PlannerAgent (word_count,
                           complexity_score, action, reason)
            chunk_index: Position of this chunk in the document (0-indexed)
            total_chunks: Total number of chunks in the document
            
        Returns:
            Dictionary with 'action', 'summary_length', and 'reason' keys
        """
        if not self.model:
            # Use PlannerAgent's complexity analysis for smarter fallback routing
            if complexity_info and complexity_info.get("complexity_score", 0) > 0.7:
                logger.info(f"Fallback mode: chunk {chunk_index+1}/{total_chunks} is complex "
                           f"(score={complexity_info['complexity_score']:.2f}), "
                           f"but no Gemini API available — routing to local.")
            
            return {
                "action": "summarize_local",
                "summary_length": "normal",
                "reason": "Fallback due to missing API key."
            }
            
        logger.info(f"Gemini Brain routing chunk {chunk_index+1}/{total_chunks}...")
        
        # Format keywords for the prompt
        keyword_context = ""
        if chunk_keywords:
            kw_list = [kw['keyword'] for kw in chunk_keywords[:10]]
            keyword_context = f"\nExtracted keywords from this chunk: {', '.join(kw_list)}\n"
        
        # Format complexity analysis for the prompt
        complexity_context = ""
        if complexity_info:
            complexity_context = (
                f"\nComplexity analysis (from PlannerAgent):\n"
                f"- Word count: {complexity_info.get('word_count', 'N/A')}\n"
                f"- Complexity score: {complexity_info.get('complexity_score', 'N/A'):.2f}\n"
                f"- Average sentence length: {complexity_info.get('avg_sentence_length', 'N/A'):.1f} words\n"
                f"- Complex word ratio: {complexity_info.get('complex_word_ratio', 'N/A'):.2f}\n"
                f"- Planner recommendation: {complexity_info.get('action', 'N/A')}\n"
            )
        
        prompt = (
            "You are the Supervisor Brain of an Agentic AI summarizing system. "
            "You are evaluating ONE CHUNK of a larger document. "
            f"This is chunk {chunk_index+1} of {total_chunks}.\n\n"
            "You have access to:\n"
            "- A local fine-tuned FLAN-T5 model: excellent at standard academic/factual summaries\n"
            "- Gemini API (yourself): handles complex reasoning, conversational text, or rewriting\n\n"
            "Analyze the chunk text, its keywords, and the complexity analysis, then decide how to process it.\n"
            "Return ONLY a valid JSON object with these keys:\n"
            "- 'action': one of ['summarize_local', 'summarize_gemini', 'rewrite_gemini']\n"
            "- 'summary_length': one of ['short', 'normal', 'long']\n"
            "- 'reason': brief explanation for your decision\n\n"
            "Rules:\n"
            "- Standard factual/academic text -> 'summarize_local' (fast, efficient)\n"
            "- Complex reasoning, multi-topic, or conversational -> 'summarize_gemini'\n"
            "- Very short or poorly written -> 'rewrite_gemini'\n"
            "- Prefer 'summarize_local' when possible (lower latency, no API cost)\n\n"
            f"Chunk text to analyze:\n\n{chunk_text[:5000]}{keyword_context}{complexity_context}"
        )
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            
            decision = json.loads(response.text)
            logger.info(f"Chunk {chunk_index+1} routed: {decision['action']} ({decision.get('reason', '')})")
            return decision
            
        except Exception as e:
            logger.error(f"Gemini routing failed for chunk {chunk_index+1}: {e}")
            return {
                "action": "summarize_local",
                "summary_length": "normal",
                "reason": f"Fallback due to Gemini error: {str(e)}"
            }

    def process_with_gemini(self, text: str, action: str, length: str) -> str:
        """
        Process a text chunk directly using Gemini when local models aren't suitable.
        
        Args:
            text: The text chunk to process
            action: The action to perform ('summarize_gemini' or 'rewrite_gemini')
            length: Desired summary length ('short', 'normal', 'long')
            
        Returns:
            Generated summary or rewritten text
        """
        if not self.model:
            return "Error: Gemini model not initialized."
            
        logger.info(f"Processing chunk with Gemini (action: {action})...")
        
        if action == "rewrite_gemini":
            prompt = f"Please rewrite and polish the following text to make it clear and professional:\n\n{text}"
        elif length == "short":
            prompt = f"Generate a very concise summary (around 50 words) of the following text:\n\n{text}"
        elif length == "long":
            prompt = f"Generate a detailed, comprehensive summary of the following text:\n\n{text}"
        else:
            prompt = f"Generate a balanced summary (around 150 words) of the following text:\n\n{text}"
            
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini processing failed: {e}")
            return f"Error during Gemini processing: {str(e)}"
