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
        2. Checks text length to determine strategy (standard vs refine)
        3. Creates an appropriate prompt based on desired summary length, using few-shot examples
        4. Uses the model to generate a summary with carefully tuned parameters
        5. Applies post-processing to improve readability and reduce repetition
        
        Args:
            text: The text content to summarize
            length: Desired summary length ('short', 'normal', 'long')
            
        Returns:
            Processed and improved summary text
        """
        # Normalize text first
        text = self._normalize_text(text)
        
        # Return empty summary for empty input
        if not text:
            return "No text content to summarize."
            
        # Strategy selection based on text length
        # FLAN-T5-small has a context limit of 512 tokens. 
        # Approx chars per token ~4. So 2000 chars is a safe upper bound.
        if len(text) > 2500:
            logger.info(f"Input text is long ({len(text)} chars). Using Refine strategy.")
            return self._generate_refine_summary(text, length)
        
        # Standard summarization with Few-Shot Prompting
        # Examples help guide the model to the desired style and format
        prompt_template = self._get_few_shot_prompt(length)
        prompt = prompt_template.format(text=text)
        
        # Configure generation parameters based on length
        max_new_tokens = {
            "short": 75,
            "normal": 200,
            "long": 350
        }.get(length, 200)
        
        min_new_tokens = {
            "short": 30,
            "normal": 100,
            "long": 200
        }.get(length, 100)
        
        # Generate summary
        logger.info(f"Generating {length} summary (standard)...")
        
        result = self.pipe(
            prompt,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
            min_length=min_new_tokens
        )
        
        summary = result[0]['generated_text']
        
        # Post-process
        cleaned_summary = self._post_process_summary(summary)
        return cleaned_summary

    def _generate_refine_summary(self, text: str, length: str) -> str:
        """
        Generate summary for long texts using a Refine (Chunk-and-Update) strategy.
        
        Algorithm:
        1. Split text into manageable chunks
        2. Summarize the first chunk
        3. For subsequent chunks, ask model to update the summary with new info
        
        Args:
            text: Long input text
            length: Desired summary length
            
        Returns:
            Refined summary covering the entire text
        """
        # Split text into chunks (approx 2000 chars each to fit in context)
        chunk_size = 2000
        overlap = 100
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
            
        logger.info(f"Split long text into {len(chunks)} chunks for refinement")
        
        # Step 1: Summarize first chunk
        current_summary = self.generate_summary(chunks[0], length)
        
        # Step 2: Refine with subsequent chunks
        for i, chunk in enumerate(chunks[1:], 1):
            logger.info(f"Refining summary with chunk {i+1}/{len(chunks)}")
            
            refine_prompt = f"""
Existing summary: {current_summary}

New context: {chunk}

Task: Update and refine the existing summary to include key information from the new context. 
Keep the summary coherent and flow well. Do not increase the length significantly unless necessary.
Updated summary:"""
            
            # Generate refined summary
            result = self.pipe(
                refine_prompt,
                do_sample=True,
                temperature=0.6, # Slightly lower temp for stability in refinement
                max_new_tokens=300, # Allow room for expansion
                min_length=100
            )
            
            current_summary = result[0]['generated_text']
            
        return self._post_process_summary(current_summary)

    def _get_few_shot_prompt(self, length: str) -> str:
        """
        Returns a prompt template with few-shot examples based on desired length.
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

        else: # normal
            return """Generate a balanced summary (approx 150 words). Capture the main points and key supporting details without unnecessary fluff.

Example:
Input: The Industrial Revolution was a period of major mechanization and innovation that began in Great Britain during the mid-18th century and early 19th century and later spread throughout much of the world. The American Industrial Revolution, sometimes referred to as the Second Industrial Revolution, began in the 1870s and continued through World War II. This era saw the mechanization of agriculture and textile manufacturing and a revolution in power, including steam ships and railroads, that effected social, cultural and economic conditions.
Summary: The Industrial Revolution, beginning in mid-18th century Britain, marked a major shift towards mechanization and innovation. It later spread globally, with the American Industrial Revolution (Second Industrial Revolution) starting in the 1870s. Key developments included the mechanization of agriculture and textiles, and breakthroughs in power like steam ships and railroads, fundamentally transforming social and economic structures.

Task:
Input: {text}
Summary:"""
    
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