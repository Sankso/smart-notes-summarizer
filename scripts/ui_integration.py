#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration with the Smart Notes Summarizer app.
This script provides functions to integrate the batch PDF processor 
with the Streamlit UI.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Add parent directory to path to import project modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import required modules
from scripts.batch_processor import BatchPDFProcessor
from agent.keyword_extractor import KeywordExtractor
from agent.section_detector import SectionDetector
from agent.executor import Executor

logger = logging.getLogger(__name__)

class SmartNotesSummarizerIntegration:
    """
    Class to integrate batch PDF processor with the Smart Notes Summarizer UI.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the integrator.
        
        Args:
            model_path: Path to the model directory (optional)
        """
        # Default to the project's model path if not provided
        if not model_path:
            model_path = os.path.join(parent_dir, "models", "lora_weights")
        
        logger.info(f"Initializing SmartNotesSummarizerIntegration")
        
        # Initialize batch processor
        self.batch_processor = BatchPDFProcessor(model_path)
        
        # Initialize the components
        self.keyword_extractor = KeywordExtractor()
        self.section_detector = SectionDetector()
        self.executor = Executor(model_path=model_path)
    
    def process_pdf(self, pdf_path: str, 
                   summary_length: str = "normal",
                   extract_keywords: bool = False,
                   detect_sections: bool = False) -> Dict[str, Any]:
        """
        Process a PDF file and return comprehensive results.
        
        Args:
            pdf_path: Path to the PDF file
            summary_length: Length of the summary ("short", "normal", "long")
            extract_keywords: Whether to extract keywords
            detect_sections: Whether to detect sections
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Basic processing via batch processor
        result = self.batch_processor.process_pdf(
            pdf_path, 
            summary_length=summary_length,
            extract_keywords=False  # We'll use our dedicated extractor
        )
        
        if not result["success"]:
            return result
        
        # Extract keywords if requested
        if extract_keywords:
            keywords = self.keyword_extractor.extract_keywords(result["text"])
            result["keywords"] = keywords
        
        # Detect sections if requested
        if detect_sections:
            sections = self.section_detector.detect_sections(result["text"])
            result["sections"] = sections
            
            # Generate section-wise summaries if sections were detected
            if sections:
                section_summaries = {}
                for section_title, section_text in sections.items():
                    section_summary = self.executor.generate_summary(
                        section_text, 
                        length=summary_length
                    )
                    section_summaries[section_title] = section_summary
                
                result["section_summaries"] = section_summaries
        
        return result

    def process_with_options(self, pdf_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a PDF with various options specified in the options dictionary.
        
        Args:
            pdf_path: Path to the PDF file
            options: Dictionary of processing options
                - summary_length: Length of summary ("short", "normal", "long")
                - extract_keywords: Whether to extract keywords
                - detect_sections: Whether to detect sections
                
        Returns:
            Dictionary with processing results
        """
        return self.process_pdf(
            pdf_path,
            summary_length=options.get("summary_length", "normal"),
            extract_keywords=options.get("extract_keywords", False),
            detect_sections=options.get("detect_sections", False)
        )