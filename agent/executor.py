#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Executor Module for Smart Notes Summarizer.

Core component responsible for summarizing a single text chunk using a fine-tuned
FLAN-T5 model with LoRA (Low-Rank Adaptation) weights. This module handles only
single-chunk summarization — the sliding-window chunking and per-chunk routing
is orchestrated by the agent.

Key functionalities:
1. Model loading with LoRA adapter support via PEFT
2. Single-chunk text summarization with configurable length
3. Advanced post-processing for quality improvement and repetition reduction
"""

import os
import re
import logging
import difflib
import torch
from typing import Optional, List
from collections import Counter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel

logger = logging.getLogger(__name__)


class Executor:
    """
    Executor for single-chunk text summarization using fine-tuned FLAN-T5.
    
    Loads a base FLAN-T5 model enhanced with LoRA adapters trained on
    summarization datasets. Designed to summarize individual text chunks
    that fit within the model's 512-token context window.
    
    Note: Long-document handling (sliding-window chunking, per-chunk routing)
    is managed by the SmartSummarizerAgent orchestrator, not this class.
    """
    
    def __init__(self,
                model_name: str = "google/flan-t5-small",
                model_path: Optional[str] = None):
        """
        Initialize the Executor with the specified model.
        
        Args:
            model_name: Base model identifier (e.g. "google/flan-t5-small")
            model_path: Path to LoRA adapter weights directory. If None,
                        will attempt to find weights in the default location.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Try to locate LoRA weights if path not explicitly provided
        if not model_path:
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(parent_dir, "models", "lora_weights")
            if os.path.exists(default_path):
                model_path = default_path
        
        logger.info(f"Initializing Executor with base model {model_name} on {self.device}")
        
        # Load the base model components
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Apply LoRA adapter weights if available
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading LoRA weights from {model_path}")
            try:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    model_path,
                    is_trainable=False
                )
                # Merge LoRA adapters into the base model so the pipeline
                # sees a standard T5ForConditionalGeneration, not PeftModel
                self.model = self.model.merge_and_unload()
                logger.info("LoRA weights loaded and merged successfully")
            except Exception as e:
                logger.warning(f"Failed to load LoRA weights: {e}")
                logger.warning("Falling back to base model")
        
        self.model.to(self.device)
        
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text before summarization to improve input quality.
        Handles whitespace, special characters, and common PDF extraction artifacts.
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text ready for summarization
        """
        if not text or not text.strip():
            return ""
            
        text = str(text)
        
        # Remove citations like [1], [1, 2], [1-3]
        text = re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*(?:-\s*\d+)?\s*\]', '', text)
        
        # Fix common PDF extraction artifacts
        text = re.sub(r'([a-z])- ([a-z])', r'\1\2', text)  # Fix hyphenation
        text = re.sub(r'(\w)\s+([.,;:])', r'\1\2', text)   # Fix spaced punctuation
        
        # Ensure all duplicate spaces (including those left by removed citations) are collapsed
        text = re.sub(r'\s{2,}', ' ', text)        
        # Remove repeating header/footer lines
        lines = text.split('\n')
        filtered_lines = []
        
        for i, line in enumerate(lines):
            is_header_footer = False
            if len(line.strip()) < 50:
                occurrences = [j for j, l in enumerate(lines) if l == line]
                if len(occurrences) > 2:
                    intervals = [occurrences[j+1] - occurrences[j] for j in range(len(occurrences)-1)]
                    if intervals and max(intervals) == min(intervals):
                        is_header_footer = True
            
            if not is_header_footer:
                filtered_lines.append(line)
        
        normalized_text = '\n'.join(filtered_lines)
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
        
        return normalized_text
        
    def generate_summary(self, text: str, length: str = "normal") -> str:
        """
        Generate a summary of a single text chunk.
        
        The input text should already be chunked to fit within the model's
        context window (~2000 chars). For long documents, the orchestrator
        handles chunking and calls this method per-chunk.
        
        Args:
            text: Text chunk to summarize (should be ≤ ~2000 chars)
            length: Desired summary length ('short', 'normal', 'long')
            
        Returns:
            Processed summary text for this chunk
        """
        text = self.normalize_text(text)
        
        if not text:
            return "No text content to summarize."
        
        # Build prompt with few-shot examples
        prompt_template = self._get_few_shot_prompt(length)
        prompt = prompt_template.format(text=text)
        
        # Configure generation parameters based on length
        max_new_tokens = {"short": 75, "normal": 200, "long": 350}.get(length, 200)
        min_new_tokens = {"short": 30, "normal": 100, "long": 200}.get(length, 100)
        
        logger.info(f"Generating {length} summary for chunk ({len(text)} chars)...")
        
        result = self.pipe(
            prompt,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
            min_length=min_new_tokens
        )
        
        summary = result[0]['generated_text']
        return self.post_process_summary(summary)

    def _get_few_shot_prompt(self, length: str) -> str:
        """
        Returns a prompt template with few-shot examples based on desired length.
        
        Args:
            length: Desired summary length ('short', 'normal', 'long')
            
        Returns:
            Prompt template string with {text} placeholder
        """
        if length == "short":
            return """Generate a concise summary (approx 50 words). Focus only on the main idea.

Example:
Input: The solar system formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. The vast majority of the system's mass is in the Sun, with the majority of the remaining mass contained in Jupiter. The four smaller inner planets, Mercury, Venus, Earth and Mars, are terrestrial planets, being primarily composed of rock and metal.
Summary: The solar system formed 4.6 billion years ago from a collapsing molecular cloud. Most mass is in the Sun, followed by Jupiter. The four inner terrestrial planets (Mercury, Venus, Earth, Mars) are rock and metal.

Task:
Input: {text}
Summary:"""

        elif length == "long":
            return """Generate a detailed and comprehensive summary (approx 250 words). Include specific facts, figures, and key details. Structure it logically.

Example:
Input: [Technical article about photosynthesis]
Summary: Photosynthesis is the process used by plants, algae, and certain bacteria to harness energy from sunlight and turn it into chemical energy. This energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water. The process releases oxygen as a byproduct. Photosynthesis limits the amount of carbon dioxide in the atmosphere and provides oxygen for other life forms. It principally occurs in chloroplasts, which contain the pigment chlorophyll. The overall reaction can be summarized as: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2. There are two stages: light-dependent reactions and the Calvin cycle.

Task:
Input: {text}
Summary:"""

        else:  # normal
            return """Generate a balanced summary (approx 150 words). Capture the main points and key supporting details without unnecessary fluff.

Example:
Input: The Industrial Revolution was a period of major mechanization and innovation that began in Great Britain during the mid-18th century and early 19th century and later spread throughout much of the world. The American Industrial Revolution, sometimes referred to as the Second Industrial Revolution, began in the 1870s and continued through World War II. This era saw the mechanization of agriculture and textile manufacturing and a revolution in power, including steam ships and railroads, that effected social, cultural and economic conditions.
Summary: The Industrial Revolution, beginning in mid-18th century Britain, marked a major shift towards mechanization and innovation. It later spread globally, with the American Industrial Revolution (Second Industrial Revolution) starting in the 1870s. Key developments included the mechanization of agriculture and textiles, and breakthroughs in power like steam ships and railroads, fundamentally transforming social and economic structures.

Task:
Input: {text}
Summary:"""
    
    def post_process_summary(self, summary: str) -> str:
        """
        Advanced post-processing for generated summaries.
        
        Applies several techniques to enhance summary quality:
        1. Near-duplicate sentence detection and removal
        2. Repetitive phrase identification and removal
        3. Numerical data de-duplication
        4. Formatting improvements for readability
        
        This method is public so the orchestrator can also use it for
        merging and deduplicating across multiple chunk summaries.
        
        Args:
            summary: The raw generated summary text
            
        Returns:
            Cleaned and improved summary text
        """
        summary = summary.strip()
        if not summary:
            return ""
            
        # Split text into sentences for analysis
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        if not sentences:
            return summary
            
        # STEP 1: Remove exact and near-duplicate sentences
        unique_sentences = []
        sentence_fingerprints = set()
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            normalized = re.sub(r'[^\w\s]', '', normalized)
            
            is_duplicate = False
            
            if normalized in sentence_fingerprints:
                is_duplicate = True
            else:
                for existing in sentence_fingerprints:
                    similarity = difflib.SequenceMatcher(None, normalized, existing).ratio()
                    if similarity > 0.8:
                        is_duplicate = True
                        break
            
            if not is_duplicate and len(normalized) > 3:
                unique_sentences.append(sentence)
                sentence_fingerprints.add(normalized)
        
        # STEP 2: Handle repetitive phrases within sentences
        processed_text = ' '.join(unique_sentences)
        
        words = processed_text.lower().split()
        repeated_phrases = []
        
        for phrase_len in range(3, 8):
            if len(words) <= phrase_len:
                continue
            phrases = [' '.join(words[i:i+phrase_len]) for i in range(len(words)-phrase_len+1)]
            phrase_counts = Counter(phrases)
            for phrase, count in phrase_counts.items():
                if count > 1 and len(phrase.split()) >= 3:
                    repeated_phrases.append(phrase)
        
        for phrase in sorted(repeated_phrases, key=len, reverse=True):
            pattern = re.escape(phrase)
            matches = list(re.finditer(pattern, processed_text, re.IGNORECASE))
            if len(matches) > 1:
                for match in reversed(matches[1:]):
                    start, end = match.span()
                    processed_text = processed_text[:start] + processed_text[end:]
        
        # STEP 3: Handle repeated numerical information
        processed_text = re.sub(r'(\d+(?:\.\d+)?(?:\s*%)?(?:\s*[a-zA-Z]+)?)[^\d\n.]+\1(?:[^\d\n.]+\1)*', r'\1', processed_text)
        
        # STEP 4: Fix formatting
        processed_text = re.sub(r'\.{2,}', '.', processed_text)
        processed_text = re.sub(r'([.!?])([A-Z])', r'\1 \2', processed_text)
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        # STEP 5: Ensure proper ending
        if processed_text and processed_text[-1] not in '.!?':
            processed_text += '.'
        
        return processed_text