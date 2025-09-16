#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Keyword extractor module for Smart Notes Summarizer.
Uses various NLP techniques to extract the most relevant keywords from text.
"""

import re
import logging
import spacy
import yake
import RAKE
from typing import List, Dict, Any, Union, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class KeywordExtractor:
    """
    Extracts the most relevant keywords from text using different methods:
    1. YAKE algorithm (Yet Another Keyword Extractor)
    2. RAKE algorithm (Rapid Automatic Keyword Extraction)
    3. TF-IDF with spaCy for NLP preprocessing
    
    Each method has strengths and weaknesses for different types of text.
    """
    
    def __init__(self, 
                language: str = "en",
                max_ngram_size: int = 3,
                deduplication_threshold: float = 0.9,
                top_n: int = 10):
        """
        Initialize the keyword extractor.
        
        Args:
            language: Language code (default: "en" for English)
            max_ngram_size: Maximum size of n-grams for keyword extraction
            deduplication_threshold: Threshold for deduplication
            top_n: Number of keywords to extract
        """
        self.language = language
        self.max_ngram_size = max_ngram_size
        self.deduplication_threshold = deduplication_threshold
        self.top_n = top_n
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model successfully")
        except Exception as e:
            logger.warning(f"Error loading spaCy model: {e}")
            self.nlp = None
            
        # Initialize YAKE extractor
        self.yake_extractor = yake.KeywordExtractor(
            lan=language, 
            n=max_ngram_size,
            dedupLim=deduplication_threshold,
            top=top_n,
            features=None
        )
        
        # Initialize RAKE
        self.rake = RAKE.Rake(RAKE.SmartStopList())
        
        logger.info(f"Keyword extractor initialized with max_ngram_size={max_ngram_size}, top_n={top_n}")

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace, special characters, etc.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def extract_keywords_yake(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract keywords using YAKE algorithm.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries with keywords and scores
        """
        try:
            keywords = self.yake_extractor.extract_keywords(text)
            # Convert to list of dicts for consistent format
            return [{'keyword': kw, 'score': score, 'method': 'YAKE'} for kw, score in keywords]
        except Exception as e:
            logger.warning(f"Error extracting keywords with YAKE: {e}")
            return []

    def extract_keywords_rake(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract keywords using RAKE algorithm.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries with keywords and scores
        """
        try:
            keywords = self.rake.run(text)
            # Sort by score and take top n
            keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:self.top_n]
            # Convert to list of dicts for consistent format
            return [{'keyword': kw, 'score': score, 'method': 'RAKE'} for kw, score in keywords]
        except Exception as e:
            logger.warning(f"Error extracting keywords with RAKE: {e}")
            return []

    def extract_keywords_tfidf(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract keywords using TF-IDF and spaCy.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries with keywords and scores
        """
        if not self.nlp:
            logger.warning("spaCy model not loaded, skipping TF-IDF extraction")
            return []
        
        try:
            # Process with spaCy
            doc = self.nlp(text)
            
            # Extract meaningful tokens (nouns, proper nouns, adjectives)
            tokens = [token.lemma_.lower() for token in doc 
                     if not token.is_stop and not token.is_punct 
                     and token.pos_ in ('NOUN', 'PROPN', 'ADJ')
                     and len(token.text) > 1]
            
            # Get the most common tokens
            counter = Counter(tokens)
            keywords = counter.most_common(self.top_n)
            
            # Convert to list of dicts for consistent format
            return [{'keyword': kw, 'score': score / max(counter.values()), 'method': 'TF-IDF'} 
                   for kw, score in keywords]
        except Exception as e:
            logger.warning(f"Error extracting keywords with TF-IDF: {e}")
            return []

    def extract_keywords(self, text: str, method: str = "combined") -> List[Dict[str, Any]]:
        """
        Extract keywords using specified method or a combination of methods.
        
        Args:
            text: Input text
            method: Method to use ('yake', 'rake', 'tfidf', 'combined')
            
        Returns:
            List of dictionaries with keywords and scores
        """
        logger.info(f"Extracting keywords using method: {method}")
        
        # Clean the text
        clean_text = self.clean_text(text)
        
        # Extract keywords using specified method
        if method.lower() == "yake":
            return self.extract_keywords_yake(clean_text)
        elif method.lower() == "rake":
            return self.extract_keywords_rake(clean_text)
        elif method.lower() == "tfidf":
            return self.extract_keywords_tfidf(clean_text)
        elif method.lower() == "combined":
            # Get keywords from all methods
            yake_keywords = self.extract_keywords_yake(clean_text)
            rake_keywords = self.extract_keywords_rake(clean_text)
            tfidf_keywords = self.extract_keywords_tfidf(clean_text)
            
            # Combine all keywords
            all_keywords = yake_keywords + rake_keywords + tfidf_keywords
            
            # Create a dictionary to track best score for each keyword
            best_keywords = {}
            for item in all_keywords:
                keyword = item['keyword']
                score = item['score']
                method = item['method']
                
                # For YAKE, lower score is better
                if method == 'YAKE':
                    score = 1 - score  # Invert score for consistency
                
                # Update if this keyword doesn't exist or has a better score
                if keyword not in best_keywords or score > best_keywords[keyword]['score']:
                    best_keywords[keyword] = {'keyword': keyword, 'score': score, 'method': method}
            
            # Convert dictionary to list and sort by score
            result = list(best_keywords.values())
            result.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top N results
            return result[:self.top_n]
        else:
            logger.warning(f"Unknown method: {method}, falling back to combined")
            return self.extract_keywords(text, "combined")

    def format_keywords(self, keywords: List[Dict[str, Any]], format_type: str = "simple") -> str:
        """
        Format keywords for display.
        
        Args:
            keywords: List of keyword dictionaries
            format_type: Format type ('simple', 'detailed', 'markdown')
            
        Returns:
            Formatted string of keywords
        """
        if not keywords:
            return "No keywords found."
        
        if format_type == "simple":
            return ", ".join([item['keyword'] for item in keywords])
        elif format_type == "detailed":
            return "\n".join([f"{item['keyword']} (Score: {item['score']:.3f}, Method: {item['method']})" 
                           for item in keywords])
        elif format_type == "markdown":
            lines = ["| Keyword | Score | Method |", "| --- | --- | --- |"]
            for item in keywords:
                lines.append(f"| {item['keyword']} | {item['score']:.3f} | {item['method']} |")
            return "\n".join(lines)
        else:
            return ", ".join([item['keyword'] for item in keywords])