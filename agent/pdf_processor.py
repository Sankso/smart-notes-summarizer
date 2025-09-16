#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF processing utilities for Smart Notes Summarizer.
Provides advanced PDF text extraction capabilities beyond basic PyPDF2 extraction.
"""

import os
import logging
import tempfile
from typing import List, Optional

import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Provides advanced PDF processing capabilities, including:
    - Standard text extraction
    - OCR for scanned documents
    - Layout analysis and preservation
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
        Extract text from a PDF file, using OCR if needed.
        
        Args:
            pdf_path: Path to the PDF file
            force_ocr: Force OCR processing even for regular PDFs
            
        Returns:
            Extracted text content
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return f"Error: PDF file not found at {pdf_path}"
        
        # Try standard extraction first unless forced to use OCR
        if not force_ocr:
            text = self._extract_with_pypdf2(pdf_path)
            
            # Check if enough text was extracted
            if self._is_extraction_sufficient(text):
                logger.info(f"Standard extraction successful for {pdf_path}")
                return text
        
        # Fall back to OCR if enabled and standard extraction failed
        if self.ocr_enabled:
            logger.info(f"Using OCR for {pdf_path} (standard extraction insufficient)")
            return self._extract_with_ocr(pdf_path)
        else:
            logger.warning(f"Standard extraction insufficient but OCR is disabled")
            return text  # Return whatever was extracted, even if insufficient
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """
        Extract text using PyPDF2.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from each page
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    text += page_text + "\n\n"
                    
                    # Log page statistics
                    if i < 3 or i == len(reader.pages) - 1:
                        char_count = len(page_text) if page_text else 0
                        logger.debug(f"Page {i+1}: {char_count} chars extracted")
                
                logger.info(f"PyPDF2 extracted {len(text)} chars from {len(reader.pages)} pages")
                return text
                
        except Exception as e:
            logger.error(f"Error in PyPDF2 extraction: {str(e)}")
            return ""
    
    def _extract_with_ocr(self, pdf_path: str) -> str:
        """
        Extract text using OCR (convert PDF to images first).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            # Convert PDF to images
            logger.info(f"Converting PDF to images (DPI: {self.dpi})")
            images = convert_from_path(
                pdf_path, 
                dpi=self.dpi,
                first_page=1,
                last_page=25  # Limit to 25 pages for performance
            )
            
            # Extract text from each image
            logger.info(f"Performing OCR on {len(images)} pages")
            text = ""
            
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang=self.language)
                text += page_text + "\n\n"
                
                # Log progress for every few pages
                if i % 5 == 0 or i == len(images) - 1:
                    logger.debug(f"OCR progress: {i+1}/{len(images)} pages")
            
            logger.info(f"OCR extracted {len(text)} chars")
            return text
            
        except Exception as e:
            logger.error(f"Error in OCR extraction: {str(e)}")
            return f"Error in OCR processing: {str(e)}"
    
    def _is_extraction_sufficient(self, text: str) -> bool:
        """
        Check if the extracted text is sufficient or likely needs OCR.
        
        Args:
            text: Extracted text
            
        Returns:
            True if extraction seems sufficient, False otherwise
        """
        # If almost no text was extracted, standard extraction failed
        if len(text.strip()) < 100:
            return False
        
        # Check for reasonable character density
        words = text.split()
        if len(words) < 20:
            return False
        
        # Check for common OCR indicators
        ocr_indicators = ['�', '□', '■', '▯', '▮']
        indicator_count = sum(text.count(indicator) for indicator in ocr_indicators)
        
        # If too many special characters, extraction likely failed
        if indicator_count > 20 or indicator_count > len(text) * 0.05:
            return False
            
        return True