#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF processing module for Smart Notes Summarizer.

Automated document pipeline to ingest, parse, and extract text from
unstructured PDFs. Uses PyMuPDF (fitz) as the primary extraction engine
with pytesseract OCR fallback for scanned documents.

Pipeline: PDF Upload → Text Extraction → Preprocessing → Clean Text
"""

import os
import re
import logging
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    PDF ingestion pipeline with multi-strategy text extraction.
    
    Strategies (in order of preference):
    1. PyMuPDF (fitz) — fast, handles most PDFs with layout preservation
    2. OCR via pytesseract — fallback for scanned/image-based PDFs
    """
    
    def __init__(self, 
                ocr_enabled: bool = True,
                language: str = 'eng',
                dpi: int = 300):
        """
        Initialize PDF processor.
        
        Args:
            ocr_enabled: Whether to use OCR for scanned documents
            language: OCR language model
            dpi: DPI for PDF to image conversion for OCR
        """
        self.ocr_enabled = ocr_enabled
        self.language = language
        self.dpi = dpi
        
        logger.info(f"PDF Processor initialized (OCR: {ocr_enabled}, Lang: {language})")
    
    def extract_text(self, pdf_path: str, force_ocr: bool = False) -> str:
        """
        Extract text from a PDF file using PyMuPDF, with OCR fallback.
        
        Args:
            pdf_path: Path to the PDF file
            force_ocr: Force OCR processing even for regular PDFs
            
        Returns:
            Extracted and preprocessed text content
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return f"Error: PDF file not found at {pdf_path}"
        
        # Try PyMuPDF extraction first unless forced to use OCR
        if not force_ocr:
            text = self._extract_with_pymupdf(pdf_path)
            
            if self._is_extraction_sufficient(text):
                logger.info(f"PyMuPDF extraction successful for {pdf_path}")
                return self._preprocess_text(text)
        
        # Fall back to OCR if enabled and standard extraction failed
        if self.ocr_enabled:
            logger.info(f"Using OCR for {pdf_path} (PyMuPDF extraction insufficient)")
            text = self._extract_with_ocr(pdf_path)
            return self._preprocess_text(text)
        else:
            logger.warning("PyMuPDF extraction insufficient but OCR is disabled")
            return self._preprocess_text(text)
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """
        Extract text using PyMuPDF (fitz).
        
        PyMuPDF provides fast, high-quality text extraction with layout
        preservation. It handles most PDF types including those with
        complex formatting, tables, and multi-column layouts.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for i, page in enumerate(doc):
                # Extract text with layout preservation
                page_text = page.get_text("text")
                text += page_text + "\n\n"
                
                if i < 3 or i == len(doc) - 1:
                    char_count = len(page_text) if page_text else 0
                    logger.debug(f"Page {i+1}: {char_count} chars extracted")
            
            doc.close()
            logger.info(f"PyMuPDF extracted {len(text)} chars from {len(doc)} pages")
            return text
                
        except Exception as e:
            logger.error(f"Error in PyMuPDF extraction: {str(e)}")
            return ""
    
    def _extract_with_ocr(self, pdf_path: str) -> str:
        """
        Extract text using OCR (convert PDF pages to images first).
        
        Falls back to this method when PyMuPDF extraction yields
        insufficient text (typically for scanned/image-based PDFs).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            logger.info(f"Converting PDF to images (DPI: {self.dpi})")
            images = convert_from_path(
                pdf_path, 
                dpi=self.dpi,
                first_page=1,
                last_page=25  # Limit to 25 pages for performance
            )
            
            logger.info(f"Performing OCR on {len(images)} pages")
            text = ""
            
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang=self.language)
                text += page_text + "\n\n"
                
                if i % 5 == 0 or i == len(images) - 1:
                    logger.debug(f"OCR progress: {i+1}/{len(images)} pages")
            
            logger.info(f"OCR extracted {len(text)} chars")
            return text
            
        except ImportError as e:
            logger.error(f"OCR dependencies not installed: {e}")
            return f"Error: OCR dependencies not available ({str(e)})"
        except Exception as e:
            logger.error(f"Error in OCR extraction: {str(e)}")
            return f"Error in OCR processing: {str(e)}"
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess extracted text to clean up common extraction artifacts.
        
        Handles:
        - Excessive whitespace and blank lines
        - Hyphenated line breaks
        - Spaced punctuation
        - Very short noise lines (page numbers, headers)
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""
        
        # Fix hyphenated line breaks (e.g., "infor-\nmation" → "information")
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # Fix spaced punctuation
        text = re.sub(r'(\w)\s+([.,;:])', r'\1\2', text)
        
        # Remove very short lines that are likely page numbers or headers
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are just numbers (page numbers)
            if stripped and not re.match(r'^\d{1,4}$', stripped):
                filtered_lines.append(line)
        
        # Collapse multiple blank lines into one
        text = '\n'.join(filtered_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _is_extraction_sufficient(self, text: str) -> bool:
        """
        Check if the extracted text is sufficient or likely needs OCR.
        
        Args:
            text: Extracted text
            
        Returns:
            True if extraction seems sufficient, False otherwise
        """
        if len(text.strip()) < 100:
            return False
        
        words = text.split()
        if len(words) < 20:
            return False
        
        # Check for common garbled-extraction indicators
        ocr_indicators = ['�', '□', '■', '▯', '▮']
        indicator_count = sum(text.count(indicator) for indicator in ocr_indicators)
        
        if indicator_count > 20 or indicator_count > len(text) * 0.05:
            return False
            
        return True