#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Section Detector for Smart Notes Summarizer.

This module is responsible for detecting sections in text content
and providing section-specific operations.
"""

import re
import logging
from typing import Dict, Any

from .executor import Executor

logger = logging.getLogger(__name__)

class SectionDetector:
    """
    Class for detecting and extracting sections from text.
    """
    
    def __init__(self):
        """Initialize the SectionDetector."""
        self.executor = Executor()
        
    def detect_and_summarize_sections(self, text: str, summary_length: str = "normal") -> Dict[str, str]:
        """
        Detect sections in the text and summarize each section separately.
        
        Args:
            text: Input text
            summary_length: Length of summaries ('short', 'normal', 'long')
            
        Returns:
            Dictionary with section names as keys and summaries as values
        """
        logger.info(f"Detecting and summarizing sections with {summary_length} length")
        
        # First detect sections
        sections = self.detect_sections(text)
        
        # Now summarize each section
        section_summaries = {}
        for section_name, section_text in sections.items():
            logger.info(f"Summarizing section: {section_name}")
            
            try:
                section_summary = self.executor.generate_summary(
                    section_text, 
                    length=summary_length
                )
                section_summaries[section_name] = section_summary
            except Exception as e:
                logger.warning(f"Error summarizing section {section_name}: {e}")
                section_summaries[section_name] = "Error: Failed to summarize this section."
                
        logger.info(f"Summarized {len(section_summaries)} sections")
        return section_summaries
    
    def detect_sections(self, text: str) -> Dict[str, str]:
        """
        Detect sections in the provided text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with section names as keys and section text as values
        """
        logger.info("Detecting sections in text")
        
        # Common section headings in academic and business documents
        section_patterns = [
            r'^(?:\d+\.\s*)?(?:introduction|abstract|overview|summary|background)(?:\:|\s*$)',
            r'^(?:\d+\.\s*)?(?:methodology|methods|approach|experimental setup|materials and methods)(?:\:|\s*$)',
            r'^(?:\d+\.\s*)?(?:results|findings|observations|outcome)(?:\:|\s*$)',
            r'^(?:\d+\.\s*)?(?:discussion|analysis|interpretation|evaluation)(?:\:|\s*$)',
            r'^(?:\d+\.\s*)?(?:conclusion|summary|future work|recommendations)(?:\:|\s*$)',
            r'^(?:\d+\.\s*)?(?:references|bibliography|citations|sources)(?:\:|\s*$)',
            r'^(?:\d+\.\s*)?(?:appendix|appendices|supplementary material)(?:\:|\s*$)'
        ]
        
        # Split text into lines
        lines = text.split('\n')
        
        # Find potential section headings
        section_headings = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if line matches any section pattern
            for pattern in section_patterns:
                if re.match(pattern, line.lower()):
                    section_headings.append((i, line))
                    break
        
        # If no sections found, try to identify by uppercase lines or numbering
        if not section_headings:
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Check if line is all uppercase and not too long (likely a heading)
                if line.isupper() and len(line) < 50:
                    section_headings.append((i, line))
                # Check if line starts with numbering like "1." or "1.1"
                elif re.match(r'^\d+\.(?:\d+\.?)?\s+[A-Z]', line) and len(line) < 70:
                    section_headings.append((i, line))
        
        # If still no sections found, create artificial sections
        if not section_headings:
            logger.info("No clear section headings found, creating artificial sections")
            # Divide text into roughly equal parts
            total_lines = len(lines)
            if total_lines > 100:  # Only for longer documents
                parts = min(5, total_lines // 50)  # Create up to 5 sections
                section_size = total_lines // parts
                for i in range(parts):
                    idx = i * section_size
                    heading = f"Section {i+1}"
                    section_headings.append((idx, heading))
        
        # Extract each section
        sections = {}
        for i in range(len(section_headings)):
            start_idx = section_headings[i][0]
            # If this is the last section, go to the end
            end_idx = section_headings[i+1][0] if i < len(section_headings) - 1 else len(lines)
            
            section_name = section_headings[i][1].strip()
            section_text = '\n'.join(lines[start_idx+1:end_idx])
            
            # Skip empty sections or very short ones
            if len(section_text.strip()) < 50:
                continue
                
            sections[section_name] = section_text
                
        logger.info(f"Detected {len(sections)} sections")
        return sections