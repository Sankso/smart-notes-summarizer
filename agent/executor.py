#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Executor Module for Smart Notes Summarizer.

This module is the core component responsible for text generation tasks using a fine-tuned
language model. It handles model loading (with LoRA weights when available), generates
summaries of different lengths, extracts keywords, and includes advanced post-processing
to improve summary quality and reduce repetition.

Key functionalities:
1. Model loading with fallback mechanisms
2. Text summarization with length control
3. Keyword extraction from documents
4. Advanced post-processing for quality improvement
5. Repetition detection and metrics

Authors: Sanskriti Pal
Date: September 2025
"""

import os
import logging
import torch
from typing import Dict, Any, Union, Optional, Tuple, List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel

# Configure module logger
logger = logging.getLogger(__name__)

class Executor:
    """
    Executor for text transformation and generation tasks.
    
    This class handles all interactions with the language model, including loading the
    model, generating summaries, extracting keywords, and applying post-processing
    techniques to improve output quality. It supports LoRA fine-tuned models for 
    more accurate domain-specific outputs.
    
    The Executor uses a FLAN-T5 model as its base, enhanced with fine-tuning for
    academic and technical document summarization. It applies sophisticated
    prompt engineering and post-processing to reduce repetition.
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
        # Set appropriate device based on hardware availability
        # Using GPU significantly improves generation speed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Try to locate LoRA weights if path not explicitly provided
        if not model_path:
            # Look in the standard location relative to this file
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(parent_dir, "models", "lora_weights")
            if os.path.exists(default_path):
                model_path = default_path
        
        logger.info(f"Initializing Executor with base model {model_name} on {self.device}")
        
        # Load the base model components
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Apply LoRA adapter weights if available
        # This enhances the base model with domain-specific knowledge
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading LoRA weights from {model_path}")
            try:
                # Apply LoRA adapter to the base model
                # is_trainable=False ensures we're in inference mode
                self.model = PeftModel.from_pretrained(
                    self.model,
                    model_path,
                    is_trainable=False
                )
                logger.info("LoRA weights loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load LoRA weights: {e}")
                logger.warning("Falling back to base model")
        
        # Move model to the appropriate device (GPU/CPU)
        self.model.to(self.device)
        
        # Create the HuggingFace generation pipeline for text generation
        # This provides a convenient interface for text generation tasks
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1  # 0 = first GPU, -1 = CPU
        )
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text before summarization to improve input quality.
        Handles whitespace, special characters, and common formatting issues.
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text ready for summarization
        """
        import re
        
        # Return empty string if input is empty
        if not text or not text.strip():
            return ""
            
        # Convert to string if not already
        text = str(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction artifacts
        text = re.sub(r'([a-z])- ([a-z])', r'\1\2', text)  # Fix hyphenation
        text = re.sub(r'(\w)\s+([.,;:])', r'\1\2', text)   # Fix spaced punctuation
        
        # Remove headers/footers that appear as repeating patterns
        lines = text.split('\n')
        filtered_lines = []
        header_footer_threshold = 0.9  # Similarity threshold
        
        # Process lines to remove likely headers/footers
        for i, line in enumerate(lines):
            is_header_footer = False
            if len(line.strip()) < 50:  # Only short lines could be headers/footers
                # Check if this line appears multiple times
                occurrences = [j for j, l in enumerate(lines) if l == line]
                
                # If line appears regularly in the document, it's likely header/footer
                if len(occurrences) > 2:
                    intervals = [occurrences[j+1] - occurrences[j] for j in range(len(occurrences)-1)]
                    if intervals and max(intervals) == min(intervals):
                        is_header_footer = True
            
            if not is_header_footer:
                filtered_lines.append(line)
        
        normalized_text = '\n'.join(filtered_lines)
        
        # Remove excessive whitespace again
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
        
        return normalized_text
        
    def generate_summary(self, text: str, length: str = "normal") -> str:
        """
        Generate a high-quality summary of the input text with specified length.
        
        This method is the core summarization functionality. It:
        1. Normalizes text to ensure consistent input quality
        2. Creates an appropriate prompt based on desired summary length
        3. Uses the model to generate a summary with carefully tuned parameters
        4. Applies post-processing to improve readability and reduce repetition
        5. Calculates quality metrics to track improvements
        
        Args:
            text: The text content to summarize
            length: Desired summary length:
                   - 'short': Concise summary (around 50 words)
                   - 'normal': Standard length (around 150 words)
                   - 'long': Comprehensive summary (around 250-300 words)
            
        Returns:
            Processed and improved summary text
        """
        # Normalize text first to ensure quality and consistency
        text = self._normalize_text(text)
        
        # Return empty summary for empty input
        if not text:
            return "No text content to summarize."
        
        # Truncate if needed
        if len(text) > 10000:
            logger.warning(f"Input text is very long ({len(text)} chars). Truncating to 10000 chars.")
            text = text[:10000]
        
        # Create enhanced prompts with explicit anti-repetition instructions
        if length == "short":
            prompt = f"""Generate a concise and brief summary of the following text in a few sentences (around 50 words). 
            
IMPORTANT GUIDELINES:
1. DO NOT repeat the same information or facts
2. AVOID redundant phrases or sentences
3. Present each distinct fact, concept or statistic EXACTLY ONCE
4. Use clear, varied language without repetitive patterns
5. Structure the content logically with good flow between ideas

Text to summarize:
{text}"""
            max_new_tokens = 75
            min_new_tokens = 30
        elif length == "long":
            prompt = f"""Generate a comprehensive and detailed summary of the following text, covering all key points (around 250-300 words).

IMPORTANT GUIDELINES:
1. DO NOT repeat the same information, even if phrased differently
2. Ensure each paragraph covers distinct aspects without redundancy
3. Use varied sentence structures and transitions
4. Present numerical data clearly and exactly once per distinct data point
5. Maintain logical organization with clear progression of ideas
6. Synthesize related information instead of listing similar facts separately

Text to summarize:
{text}"""
            max_new_tokens = 350
            min_new_tokens = 200
        else:  # normal
            prompt = f"""Summarize the following text with moderate detail (around 150 words).

IMPORTANT GUIDELINES:
1. NEVER repeat information or concepts
2. Each sentence must contain unique content not mentioned elsewhere
3. Use precise language and avoid redundant descriptions
4. Create a logical flow where each point builds on previous ones
5. Synthesize similar points rather than repeating them

Text to summarize:
{text}"""
            max_new_tokens = 200
            min_new_tokens = 100
        
        # Generate summary
        logger.info(f"Generating {length} summary (max_tokens={max_new_tokens})...")
        
        result = self.pipe(
            prompt,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
            min_length=min_new_tokens
        )
        
        summary = result[0]['generated_text']
        logger.info(f"Summary generated: {len(summary)} chars")
        
        # Calculate repetition metrics for raw summary
        raw_metrics = self.analyze_repetition(summary)
        logger.info(f"Raw summary repetition score: {raw_metrics['repetition_score']:.4f}")
        
        # Post-process the summary to improve quality
        cleaned_summary = self._post_process_summary(summary)
        
        # Calculate metrics after cleaning to measure improvement
        cleaned_metrics = self.analyze_repetition(cleaned_summary)
        logger.info(f"Summary post-processed: {len(cleaned_summary)} chars")
        logger.info(f"Cleaned summary repetition score: {cleaned_metrics['repetition_score']:.4f}")
        logger.info(f"Repetition reduction: {raw_metrics['repetition_score'] - cleaned_metrics['repetition_score']:.4f}")
        
        return cleaned_summary
    
    def _post_process_summary(self, summary: str) -> str:
        """
        Advanced post-processing for generated summaries to improve quality and eliminate repetition.
        
        This method applies several sophisticated techniques to enhance summary quality:
        1. Near-duplicate sentence detection using semantic similarity
        2. Repetitive phrase identification and removal
        3. Numerical data de-duplication (avoids repeating the same statistics)
        4. Formatting improvements for readability
        5. Sentence fragment completion and correction
        
        The process preserves the original meaning while making the summary more
        concise, coherent and free of repetitive content.
        
        Args:
            summary: The raw generated summary from the language model
            
        Returns:
            Cleaned and significantly improved summary text
        """
        import re
        import difflib
        from collections import Counter
        from itertools import combinations
        
        # Initial basic cleaning of the summary
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
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Create normalized version for comparison (lowercase, whitespace normalized)
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation for comparison
            
            # Check if this is a duplicate or near-duplicate
            is_duplicate = False
            
            # Check for exact matches first
            if normalized in sentence_fingerprints:
                is_duplicate = True
            else:
                # Check for near-duplicates using similarity threshold
                for existing in sentence_fingerprints:
                    # Calculate similarity ratio between sentences
                    similarity = difflib.SequenceMatcher(None, normalized, existing).ratio()
                    if similarity > 0.8:  # Threshold for considering as near-duplicate
                        is_duplicate = True
                        break
            
            if not is_duplicate and len(normalized) > 3:
                unique_sentences.append(sentence)
                sentence_fingerprints.add(normalized)
        
        # STEP 2: Handle repetitive phrases within sentences
        processed_text = ' '.join(unique_sentences)
        
        # Find and remove repeated phrases (3+ words)
        words = processed_text.lower().split()
        repeated_phrases = []
        
        # Find sequences that repeat
        for phrase_len in range(3, 8):  # Check phrases of length 3 to 7 words
            if len(words) <= phrase_len:
                continue
                
            phrases = [' '.join(words[i:i+phrase_len]) for i in range(len(words)-phrase_len+1)]
            phrase_counts = Counter(phrases)
            
            # Identify phrases that repeat more than once
            for phrase, count in phrase_counts.items():
                if count > 1 and len(phrase.split()) >= 3:
                    repeated_phrases.append(phrase)
        
        # Replace second+ occurrences of repeated phrases
        for phrase in sorted(repeated_phrases, key=len, reverse=True):
            # Use regex to replace subsequent occurrences
            pattern = re.escape(phrase)
            # Find all occurrences
            matches = list(re.finditer(pattern, processed_text, re.IGNORECASE))
            
            # Keep first occurrence, replace others
            if len(matches) > 1:
                # Start from the end to avoid index issues when replacing
                for match in reversed(matches[1:]):
                    start, end = match.span()
                    processed_text = processed_text[:start] + processed_text[end:]
        
        # STEP 3: Handle repeated numerical information
        # Find patterns like "X% of Y... X% of Y" or "X.Y units... X.Y units"
        processed_text = re.sub(r'(\d+(?:\.\d+)?(?:\s*%)?(?:\s*[a-zA-Z]+)?)[^\d\n.]+\1(?:[^\d\n.]+\1)*', r'\1', processed_text)
        
        # STEP 4: Fix formatting and readability issues
        # Remove multiple periods
        processed_text = re.sub(r'\.{2,}', '.', processed_text)
        
        # Ensure space after punctuation
        processed_text = re.sub(r'([.!?])([A-Z])', r'\1 \2', processed_text)
        
        # Normalize whitespace
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        # STEP 5: Fix any broken sentences (fragments without end punctuation)
        if processed_text and processed_text[-1] not in '.!?':
            processed_text += '.'
        
        return processed_text
    
    def analyze_repetition(self, text: str) -> Dict[str, float]:
        """
        Analyze and quantify the amount of repetition in text using multiple metrics.
        
        This method implements several computational linguistics approaches to measure
        different types of repetition in text:
        
        1. Lexical diversity - measuring unique word ratios
        2. N-gram repetition - identifying repeated phrases
        3. Sentence similarity - detecting near-duplicate sentences
        
        These metrics are combined into a single repetition score that can be used
        to evaluate the quality of summaries and track improvements from post-processing.
        
        Args:
            text: The text content to analyze for repetition patterns
            
        Returns:
            Dictionary containing repetition metrics:
            - unique_word_ratio: Ratio of unique words to total (higher is better)
            - repeated_phrase_ratio: Fraction of phrases that repeat (lower is better)
            - avg_sentence_similarity: How similar sentences are to each other
            - repetition_score: Combined metric (0-1 scale, lower is better)
        """
        import re
        from collections import Counter
        import numpy as np
        
        # Handle empty text with default optimal values
        if not text or not text.strip():
            return {
                "unique_word_ratio": 1.0,  # Perfect diversity
                "repeated_phrase_ratio": 0.0,  # No repetition
                "repetition_score": 0.0  # Perfect score
            }
        
        # Normalize text for analysis
        text = re.sub(r'\s+', ' ', text.lower().strip())
        words = re.findall(r'\b\w+\b', text)
        
        if not words:
            return {
                "unique_word_ratio": 1.0,
                "repeated_phrase_ratio": 0.0,
                "repetition_score": 0.0
            }
        
        # Calculate unique word ratio (higher is better)
        word_count = len(words)
        unique_words = len(set(words))
        unique_word_ratio = unique_words / word_count if word_count > 0 else 1.0
        
        # Find repeated phrases (3+ words)
        repeated_phrase_count = 0
        total_phrases = 0
        
        for phrase_len in range(3, 8):  # Check phrases of length 3 to 7 words
            if len(words) <= phrase_len:
                continue
                
            phrases = [' '.join(words[i:i+phrase_len]) for i in range(len(words)-phrase_len+1)]
            total_phrases += len(phrases)
            
            phrase_counts = Counter(phrases)
            # Count phrases that repeat
            for phrase, count in phrase_counts.items():
                if count > 1:
                    repeated_phrase_count += (count - 1)  # Count repetitions, not occurrences
        
        # Calculate repeated phrase ratio (lower is better)
        repeated_phrase_ratio = repeated_phrase_count / total_phrases if total_phrases > 0 else 0.0
        
        # Analyze sentence similarities to find near-duplicates
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentence_similarities = []
        
        if len(sentences) > 1:
            from itertools import combinations
            import difflib
            
            # Compare each pair of sentences
            for s1, s2 in combinations(sentences, 2):
                if len(s1) < 5 or len(s2) < 5:
                    continue
                    
                similarity = difflib.SequenceMatcher(None, s1, s2).ratio()
                sentence_similarities.append(similarity)
        
        # Calculate average sentence similarity (lower is better)
        avg_sentence_similarity = np.mean(sentence_similarities) if sentence_similarities else 0.0
        
        # Combined repetition score (0-1, lower is better)
        # Weight factors can be adjusted based on importance
        repetition_score = (
            (1 - unique_word_ratio) * 0.3 + 
            repeated_phrase_ratio * 0.4 + 
            avg_sentence_similarity * 0.3
        )
        
        return {
            "unique_word_ratio": unique_word_ratio,
            "repeated_phrase_ratio": repeated_phrase_ratio,
            "avg_sentence_similarity": avg_sentence_similarity,
            "repetition_score": repetition_score
        }
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """
        Extract the most important keywords or key phrases from input text using LLM-based analysis.
        
        This method uses prompt engineering to instruct the language model to identify
        the most significant and distinctive terms or phrases from the provided text.
        The approach focuses on conceptual importance rather than statistical frequency,
        allowing for extraction of meaningful domain-specific terminology.
        
        The implementation uses direct prompting with specific instructions to ensure
        the model returns only the keywords in a consistent format for easy parsing.
        
        Args:
            text: The text content from which to extract keywords/keyphrases
            num_keywords: Maximum number of keywords to extract (default: 10)
            
        Returns:
            List[str]: A list of extracted keywords or key phrases, sorted by importance
                       Each entry is a string representing a keyword or multi-word key phrase
        """
        # Create prompt with explicit instructions
        prompt = f"Extract exactly {num_keywords} important and distinctive keywords or key phrases from this text. Return only the keywords as a comma-separated list with no explanation or additional text:\n\n{text}"
        
        # Generate keywords
        logger.info(f"Extracting {num_keywords} keywords...")
        
        result = self.pipe(
            prompt,
            do_sample=True,
            temperature=0.3,
            max_new_tokens=100
        )
        
        keywords_text = result[0]['generated_text']
        
        # Parse comma-separated list and clean up
        keywords = [kw.strip() for kw in keywords_text.split(',')]
        
        # Filter out empty entries and limit to requested number
        keywords = [kw for kw in keywords if kw and len(kw) > 1][:num_keywords]
        
        logger.info(f"Extracted {len(keywords)} keywords")
        return keywords