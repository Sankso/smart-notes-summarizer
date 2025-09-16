#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script that demonstrates how to use the batch processing capabilities.

This script provides examples of using:
1. The batch_processor.py module for programmatic PDF processing
2. The cli_app.py for command-line PDF processing
3. The ui_integration.py for integration with the Streamlit UI
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import project modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the modules
from scripts.batch_processor import BatchPDFProcessor
from scripts.ui_integration import SmartNotesSummarizerIntegration

def example_batch_processor():
    """Example of using the BatchPDFProcessor directly."""
    print("=== Example: Using BatchPDFProcessor directly ===")
    
    # Initialize the processor
    processor = BatchPDFProcessor()
    
    # Define a PDF path (replace with an actual PDF path)
    pdf_path = "path/to/your/document.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        print("Please specify a valid PDF path to run this example")
        return
    
    # Process a single PDF with default settings
    print(f"Processing PDF: {pdf_path}")
    result = processor.process_pdf(pdf_path)
    
    if result["success"]:
        print("Summary:")
        print(result["summary"])
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Process the same PDF with custom settings
    print("\nProcessing with custom settings (short summary):")
    result = processor.process_pdf(pdf_path, summary_length="short", extract_keywords=True)
    
    if result["success"]:
        print("Short Summary:")
        print(result["summary"])
        
        if "keywords" in result:
            print("\nKeywords:")
            print(result["keywords"])
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

def example_ui_integration():
    """Example of using the UI integration module."""
    print("=== Example: Using SmartNotesSummarizerIntegration ===")
    
    # Initialize the integration
    integration = SmartNotesSummarizerIntegration()
    
    # Define a PDF path (replace with an actual PDF path)
    pdf_path = "path/to/your/document.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        print("Please specify a valid PDF path to run this example")
        return
    
    # Process a PDF with all options enabled
    print(f"Processing PDF with all options enabled: {pdf_path}")
    result = integration.process_pdf(
        pdf_path,
        summary_length="normal",
        extract_keywords=True,
        detect_sections=True
    )
    
    if result["success"]:
        print("Summary:")
        print(result["summary"])
        
        if "keywords" in result and result["keywords"]:
            print("\nKeywords:")
            print(", ".join(result["keywords"]))
        
        if "sections" in result and result["sections"]:
            print("\nDetected Sections:")
            for section_title in result["sections"].keys():
                print(f"- {section_title}")
        
        if "section_summaries" in result and result["section_summaries"]:
            print("\nSection Summaries:")
            for title, summary in result["section_summaries"].items():
                print(f"\n## {title}")
                print(summary)
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

def example_cli_usage():
    """Show examples of CLI usage (no actual execution)."""
    print("=== Example: CLI Usage Examples ===")
    
    print("To process a single PDF file:")
    print("python -m scripts.cli_app --pdf_path path/to/document.pdf")
    
    print("\nTo process all PDFs in a directory:")
    print("python -m scripts.cli_app --pdf_path path/to/pdf_directory --mode batch")
    
    print("\nTo generate a short summary with keywords:")
    print("python -m scripts.cli_app --pdf_path path/to/document.pdf --summary_length short --extract_keywords")
    
    print("\nTo save results in a custom directory:")
    print("python -m scripts.cli_app --pdf_path path/to/document.pdf --output_dir path/to/output_directory")

if __name__ == "__main__":
    print("Smart Notes Summarizer - Batch Processing Examples")
    print("=================================================")
    
    # Run the examples
    example_batch_processor()
    print("\n")
    example_ui_integration()
    print("\n")
    example_cli_usage()