#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Planner agent for Smart Notes Summarizer.
Decides whether input requires summarization or rewriting based on length and complexity.
"""

import re
import math
import logging
import nltk
from typing import Dict, Any, Tuple

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

class PlannerAgent:
    """
    Planner agent that determines the best approach to process the input text.
    Based on text length, complexity, and structure, it decides whether to:
    1. Generate a summary (for long documents)
    2. Rewrite/clarify (for short texts that need refinement)
    """
    
    def __init__(self, 
                 long_text_threshold: int = 500,
                 complexity_threshold: float = 0.65):
        """
        Initialize the planner agent with thresholds for decision making.
        
        Args:
            long_text_threshold: Word count threshold to consider text as "long"
            complexity_threshold: Text complexity threshold (0-1)
        """
        self.long_text_threshold = long_text_threshold
        self.complexity_threshold = complexity_threshold
        logger.info(f"Planner agent initialized with: long_text_threshold={long_text_threshold}, "
                   f"complexity_threshold={complexity_threshold}")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze the input text and determine the best processing strategy.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing analysis results and recommended action
        """
        # Extract text statistics
        word_count, avg_sentence_length, complex_word_ratio = self._get_text_statistics(text)
        
        # Calculate complexity score (weighted combination of metrics)
        complexity_score = self._calculate_complexity_score(word_count, avg_sentence_length, complex_word_ratio)
        
        # Determine if text needs summarization or rewriting
        needs_summarization = (word_count > self.long_text_threshold) or \
                             (complexity_score > self.complexity_threshold)
        
        # Create analysis result
        result = {
            "word_count": word_count,
            "avg_sentence_length": avg_sentence_length,
            "complex_word_ratio": complex_word_ratio,
            "complexity_score": complexity_score,
            "action": "summarize" if needs_summarization else "rewrite",
            "reason": self._get_reason(word_count, complexity_score, needs_summarization)
        }
        
        logger.info(f"Planner analysis: {result['action']} (score: {complexity_score:.2f}, words: {word_count})")
        return result
    
    def _get_text_statistics(self, text: str) -> Tuple[int, float, float]:
        """
        Extract statistical features from the text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (word_count, avg_sentence_length, complex_word_ratio)
        """
        # Clean and normalize text
        cleaned_text = re.sub(r'[^\w\s\.]', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Tokenize text into sentences and words
        sentences = nltk.sent_tokenize(cleaned_text)
        words = nltk.word_tokenize(cleaned_text)
        
        # Calculate word count
        word_count = len(words)
        
        # Calculate average sentence length
        avg_sentence_length = word_count / max(1, len(sentences))
        
        # Calculate complex word ratio (words with 3+ syllables)
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
        complex_word_ratio = complex_words / max(1, word_count)
        
        return word_count, avg_sentence_length, complex_word_ratio
    
    def _count_syllables(self, word: str) -> int:
        """
        Count the number of syllables in a word using a simple heuristic.
        
        Args:
            word: Input word
            
        Returns:
            Estimated syllable count
        """
        # Convert to lowercase
        word = word.lower()
        
        # Special cases
        if len(word) <= 3:
            return 1
        
        # Remove trailing e
        if word.endswith('e'):
            word = word[:-1]
        
        # Count vowel groups
        vowels = 'aeiouy'
        count = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        
        # Ensure at least one syllable
        return max(1, count)
    
    def _calculate_complexity_score(self, 
                                   word_count: int, 
                                   avg_sentence_length: float, 
                                   complex_word_ratio: float) -> float:
        """
        Calculate a complexity score based on text statistics.
        
        Args:
            word_count: Total number of words
            avg_sentence_length: Average sentence length
            complex_word_ratio: Ratio of complex words
            
        Returns:
            Complexity score between 0-1
        """
        # Normalize metrics to 0-1 range
        norm_word_count = min(1.0, word_count / (2 * self.long_text_threshold))
        norm_sentence_length = min(1.0, avg_sentence_length / 30.0)  # 30 words is quite long
        
        # Weighted combination
        score = (0.4 * norm_word_count + 
                 0.3 * norm_sentence_length + 
                 0.3 * complex_word_ratio)
        
        return score
    
    def _get_reason(self, 
                   word_count: int, 
                   complexity_score: float, 
                   needs_summarization: bool) -> str:
        """
        Generate a human-readable reason for the decision.
        
        Args:
            word_count: Total number of words
            complexity_score: Calculated complexity score
            needs_summarization: Whether summarization is needed
            
        Returns:
            String explaining the decision
        """
        if needs_summarization:
            if word_count > self.long_text_threshold:
                return f"Text is lengthy ({word_count} words > threshold of {self.long_text_threshold})"
            else:
                return f"Text is complex (complexity score {complexity_score:.2f} > threshold of {self.complexity_threshold})"
        else:
            return f"Text is relatively short ({word_count} words) and simple (complexity score {complexity_score:.2f})"