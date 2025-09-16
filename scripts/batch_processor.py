#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration module for using the PDF batch processor with the Smart Notes Summarizer.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path to import project modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the PDF processor
from scripts.process_pdfs import (
    load_model,
    extract_text_from_pdf,
    chunk_text,
    summarize_chunk,
    combine_chunk_summaries,
    evaluate_summary
)

logger = logging.getLogger(__name__)

class BatchPDFProcessor:
    """
    Class to process PDF files in batch mode using the fine-tuned model.
    Integrates with the rest of the Smart Notes Summarizer application.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the batch PDF processor.
        
        Args:
            model_path: Path to the model directory (optional)
        """
        # Default to the project's model path if not provided
        if not model_path:
            model_path = os.path.join(parent_dir, "models", "lora_weights")
        
        logger.info(f"Initializing BatchPDFProcessor with model from {model_path}")
        self.tokenizer, self.model = load_model(model_path)
    
    def process_pdf(self, pdf_path: str, 
                   summary_length: str = "normal",
                   extract_keywords: bool = False) -> Dict[str, Any]:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            summary_length: Length of the summary ("short", "normal", "long")
            extract_keywords: Whether to extract keywords
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing PDF: {pdf_path} with summary_length={summary_length}")
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        if not text or "Error extracting text" in text:
            return {
                "success": False,
                "summary": "",
                "error": text if "Error extracting text" in text else "No text extracted from PDF"
            }
        
        # Split into chunks and summarize
        chunks = chunk_text(text, self.tokenizer)
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
            
            # Pass the summary_length parameter to control output length
            summary = summarize_chunk(chunk, self.tokenizer, self.model, summary_length=summary_length)
            chunk_summaries.append(summary)
        
        # Combine summaries
        final_summary = combine_chunk_summaries(chunk_summaries)
        
        result = {
            "success": True,
            "summary": final_summary,
            "text": text
        }
        
        # Extract keywords if requested
        if extract_keywords:
            try:
                # Import the keyword extractor here to avoid circular imports
                from agent.keyword_extractor import KeywordExtractor
                
                # Create a keyword extractor and extract keywords
                keyword_extractor = KeywordExtractor(top_n=15)  # Extract more keywords for better coverage
                extracted_keywords = keyword_extractor.extract_keywords(text, "combined")
                
                # Get just the keyword strings
                result["keywords"] = [item['keyword'] for item in extracted_keywords]
            except Exception as e:
                logger.warning(f"Error extracting keywords: {e}")
                result["keywords"] = []
            
        return result
    
    def process_directory(self, pdf_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        Process all PDFs in a directory.
        
        Args:
            pdf_dir: Directory containing PDF files
            
        Returns:
            Dictionary mapping filenames to results
        """
        logger.info(f"Processing all PDFs in directory: {pdf_dir}")
        
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        results = {}
        
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
            result = self.process_pdf(pdf_file)
            results[filename] = result
        
        return results