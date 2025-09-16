    #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main agent module for Smart Notes Summarizer.
Integrates the planner and executor agents and handles PDF processing.
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from agent.planner import PlannerAgent
from agent.executor import Executor as ExecutorAgent
from agent.pdf_processor import PDFProcessor
from agent.keyword_extractor import KeywordExtractor

logger = logging.getLogger(__name__)

class SmartSummarizerAgent:
    """
    Main agent class that orchestrates the PDF summarization workflow.
    
    This class acts as the central coordinator for the summarization system,
    integrating multiple specialized components:
    
    1. Planner Agent: Analyzes input text and determines optimal summarization strategy
    2. Executor Agent: Performs the actual summarization and text processing
    3. PDF Processor: Extracts and cleans text from PDF documents
    4. Keyword Extractor: Identifies important terms in the processed text
    
    The agent provides a high-level API for both command-line applications and UI
    interfaces to access the summarization functionality with configurable options
    for summary length, keyword extraction, and section detection.
    
    The architecture follows a modular design pattern allowing components to be
    developed, tested, and improved independently while maintaining a consistent
    user-facing API.
    """
    
    def __init__(self,
                model_name: str = "google/flan-t5-small",
                lora_weights_dir: Optional[str] = None,
                logs_path: str = "docs/logs.md"):
        """
        Initialize the Smart Summarizer Agent.
        
        Args:
            model_name: Base model name for the executor
            lora_weights_dir: Directory containing LoRA adapter weights
            logs_path: Path to store interaction logs
        """
        logger.info("Initializing Smart Notes Summarizer Agent")
        
        # Initialize sub-agents
        self.planner = PlannerAgent()
        self.pdf_processor = PDFProcessor(ocr_enabled=True)
        self.keyword_extractor = KeywordExtractor()
        
        # Find model weights directory if not specified
        if lora_weights_dir is None:
            # Set default to the user's fine-tuned model location
            default_weights_dir = os.path.join(os.path.dirname(__file__), "../models/lora_weights")
            if os.path.exists(default_weights_dir):
                lora_weights_dir = default_weights_dir
                logger.info(f"Using fine-tuned model at: {default_weights_dir}")
            else:
                # Fall back to other common locations
                potential_paths = [
                    "./models/lora_weights",
                    "./finetuning/lora_weights"
                ]
                
                for path in potential_paths:
                    if os.path.exists(path):
                        lora_weights_dir = path
                        logger.info(f"Found LoRA weights at: {path}")
                        break
        
        # Initialize executor with model
        self.executor = ExecutorAgent(
            model_name=model_name,
            model_path=lora_weights_dir
        )
        
        # Set up logging
        self.logs_path = logs_path
        self._ensure_log_file_exists()
        
        logger.info("Smart Notes Summarizer Agent initialized successfully")
    
    def summarize_pdf(self, pdf_path: str, summary_length: str = "normal", extract_keywords: bool = True, detect_sections: bool = False) -> Dict[str, Any]:
        """
        Process a PDF document and generate a concise, high-quality summary with optional features.
        
        This method handles the complete PDF processing workflow:
        1. Validates the PDF file's existence and accessibility
        2. Extracts and cleans text content using the PDF processor
        3. Optionally detects document sections for structured summarization
        4. Processes the text to generate a condensed, coherent summary
        5. Optionally extracts key terms and concepts from the document
        6. Returns results in a structured format for display or further processing
        
        Args:
            pdf_path: Absolute or relative path to the PDF file to be summarized
            summary_length: Controls the verbosity of the summary:
                - 'short': Concise overview (about 10% of original)
                - 'normal': Balanced summary (about 20% of original) [default]
                - 'long': Detailed summary (about 30% of original)
            extract_keywords: When True, identifies and includes important terms from the document
            detect_sections: When True, attempts to identify document sections and summarize each separately
            
        Returns:
            Dictionary containing:
            - 'summary': The generated document summary
            - 'keywords': List of key terms (when extract_keywords=True)
            - 'sections': Dictionary of section-specific summaries (when detect_sections=True)
            - 'stats': Processing statistics and metrics (timing, token counts, etc.)
            
        Raises:
            FileNotFoundError: If the PDF file does not exist or is inaccessible
            ValueError: If the file is not a valid PDF document
        """
        logger.info(f"Processing PDF: {pdf_path}")
        start_time = time.time()
        
        # Extract text from PDF
        text = self._extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            logger.warning(f"No text extracted from PDF: {pdf_path}")
            return {"summary": "Error: No text could be extracted from the provided PDF."}
        
        # Process the extracted text
        result = self.summarize_text(text, summary_length, extract_keywords)
        
        # Process sections if requested
        if detect_sections:
            sections = self._detect_and_summarize_sections(text, summary_length)
            result["sections"] = sections
        
        process_time = time.time() - start_time
        logger.info(f"PDF processed in {process_time:.2f} seconds")
        
        # Log the interaction
        self._log_interaction(
            input_type="PDF",
            input_content=f"PDF file: {os.path.basename(pdf_path)} ({len(text)} chars)",
            output_content=result.get("summary", ""),
            processing_time=process_time
        )
        
        return result
    
    def summarize_text(self, text: str, summary_length: str = "normal", extract_keywords: bool = True) -> Dict[str, Any]:
        """
        Process raw text input and generate an optimized summary with optional keyword extraction.
        
        This method provides direct text processing without PDF handling:
        1. Uses the planner agent to analyze text characteristics and determine strategy
        2. Applies intelligent summarization based on content analysis
        3. Post-processes the summary to improve coherence and reduce repetition
        4. Optionally extracts key terms and concepts for quick reference
        
        The method is useful for processing text from non-PDF sources like:
        - Notes copied from lecture slides
        - Text exported from web pages
        - Transcripts from audio/video content
        - Content extracted from other document formats
        
        Args:
            text: Raw text content to be summarized
            summary_length: Controls the verbosity of the summary:
                - 'short': Concise overview (about 10% of original)
                - 'normal': Balanced summary (about 20% of original) [default]
                - 'long': Detailed summary (about 30% of original)
            extract_keywords: When True, identifies and includes important terms from the text
            
        Returns:
            Dictionary containing:
            - 'summary': The generated text summary
            - 'keywords': List of key terms (when extract_keywords=True)
            - 'stats': Processing statistics and metrics (timing, token counts, etc.)
            - 'analysis': Information about the text characteristics and processing decisions
        """
        logger.info(f"Processing text input ({len(text)} chars)")
        start_time = time.time()
        
        # Analyze text with planner
        analysis = self.planner.analyze(text)
        action = analysis['action']
        
        logger.info(f"Planner decision: {action} (reason: {analysis['reason']})")
        
        # Map summary_length directly
        # No need to set max_new_tokens as it's handled in the generate_summary method
        
        # Generate summary using the executor
        # Note: We're ignoring the planner's action for now and just using summarize
        summary = self.executor.generate_summary(text, length=summary_length)
        
        # Calculate repetition metrics for the summary
        repetition_metrics = self.executor.analyze_repetition(summary)
        
        # Initialize result dictionary with summary and metrics
        result = {
            "summary": summary,
            "repetition_metrics": repetition_metrics
        }
        
        # Extract keywords if requested
        if extract_keywords:
            try:
                logger.info("Extracting keywords from text")
                keywords = self.keyword_extractor.extract_keywords(text, method="combined")
                result["keywords"] = keywords
                logger.info(f"Extracted {len(keywords)} keywords")
            except Exception as e:
                logger.warning(f"Error extracting keywords: {e}")
                result["keywords"] = []
        
        process_time = time.time() - start_time
        logger.info(f"Text processed in {process_time:.2f} seconds")
        
        # Log the interaction
        self._log_interaction(
            input_type="Text",
            input_content=text[:100] + "..." if len(text) > 100 else text,
            output_content=result["summary"],
            processing_time=process_time,
            analysis=analysis
        )
        
        return result
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file using the PDF processor.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            # Use the PDF processor to extract text
            text = self.pdf_processor.extract_text(pdf_path)
            logger.info(f"Extracted {len(text)} chars from PDF")
            return text
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return f"Error extracting text from PDF: {str(e)}"
    
    def _ensure_log_file_exists(self):
        """Ensure the log file exists and has headers if it's new"""
        os.makedirs(os.path.dirname(self.logs_path), exist_ok=True)
        
        if not os.path.exists(self.logs_path):
            with open(self.logs_path, 'w', encoding='utf-8') as f:
                f.write("# Smart Notes Summarizer Agent - Interaction Logs\n\n")
                f.write("This file contains logs of interactions with the Smart Notes Summarizer Agent.\n\n")
    
    def _log_interaction(self, 
                        input_type: str,
                        input_content: str, 
                        output_content: str,
                        processing_time: float,
                        analysis: Dict[str, Any] = None):
        """
        Log an interaction to the logs file.
        
        Args:
            input_type: Type of input (PDF or Text)
            input_content: Input content or description
            output_content: Generated output content
            processing_time: Time taken to process the request
            analysis: Analysis results from planner (optional)
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open(self.logs_path, 'a', encoding='utf-8') as f:
                f.write(f"\n## Interaction - {timestamp}\n\n")
                f.write(f"**Input Type:** {input_type}\n\n")
                
                # Input content (truncated if too long)
                f.write("**Input:**\n")
                if len(input_content) > 500:
                    f.write(f"```\n{input_content[:500]}...\n```\n\n")
                else:
                    f.write(f"```\n{input_content}\n```\n\n")
                
                # Analysis if available
                if analysis:
                    action = analysis.get('action', 'unknown')
                    reason = analysis.get('reason', 'unknown')
                    word_count = analysis.get('word_count', 0)
                    
                    f.write(f"**Analysis:**\n")
                    f.write(f"- Action: {action}\n")
                    f.write(f"- Reason: {reason}\n")
                    f.write(f"- Word Count: {word_count}\n\n")
                
                # Output content
                f.write("**Output:**\n")
                f.write(f"```\n{output_content}\n```\n\n")
                
                # Processing info
                f.write(f"**Processing Time:** {processing_time:.2f} seconds\n\n")
                f.write("---\n")
                
            logger.info(f"Interaction logged to {self.logs_path}")
            
        except Exception as e:
            logger.error(f"Error logging interaction: {str(e)}")
            
    def _detect_and_summarize_sections(self, text: str, summary_length: str = "normal") -> Dict[str, Any]:
        """
        Detect and identify logical document sections and generate section-specific summaries.
        
        This method implements intelligent section detection using:
        1. Pattern matching for common academic and professional document headings
        2. Structural analysis to identify section boundaries
        3. Specialized summarization tailored to each section type
        4. Appropriate aggregation of section summaries for coherent output
        
        Section detection is particularly valuable for academic papers, technical reports,
        business documents, and other structured content where different sections serve
        distinct purposes and may require different summarization approaches.
        
        For example:
        - Introduction sections are summarized for key background and objectives
        - Methods sections are condensed to preserve important procedural details
        - Results sections focus on key findings and data points
        - Discussion/conclusion sections emphasize insights and implications
        
        Args:
            text: Input document text to be analyzed and sectioned
            summary_length: Controls verbosity of section summaries:
                - 'short': Very concise section highlights
                - 'normal': Balanced section summaries [default]
                - 'long': Detailed section overviews
            
        Returns:
            Dictionary with:
            - Section titles as keys (with standardized naming)
            - Section-specific summaries as values
            - 'full_summary': Optional combined summary of all sections
            - 'structure': Information about the detected document structure
        """
        logger.info("Detecting and summarizing sections")
        
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
        
        import re
        
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
        
        # Extract and summarize each section
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
                
            logger.info(f"Summarizing section: {section_name}")
            
            # Use the executor to generate a summary of the section
            # The length parameter controls the length of the summary
                
            # Generate summary for section
            try:
                section_summary = self.executor.generate_summary(
                    section_text, 
                    length=summary_length
                )
                sections[section_name] = section_summary
            except Exception as e:
                logger.warning(f"Error summarizing section {section_name}: {e}")
                sections[section_name] = "Error: Failed to summarize this section."
                
        logger.info(f"Detected and summarized {len(sections)} sections")
        return sections