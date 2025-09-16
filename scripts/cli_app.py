#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for the Smart Notes Summarizer.
Provides batch and single file PDF processing capabilities.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import project modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the BatchPDFProcessor
from scripts.batch_processor import BatchPDFProcessor

# Configure logging
def setup_logging(output_dir):
    """Set up logging configuration"""
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"summarizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Smart Notes Summarizer CLI")
    
    parser.add_argument("--pdf_path", required=True, 
                      help="Path to a PDF file or directory containing PDFs")
    
    parser.add_argument("--mode", choices=["single", "batch"], default="single",
                      help="Processing mode: single file or batch directory (default: single)")
    
    parser.add_argument("--output_dir", default=os.path.join(parent_dir, "exports"),
                      help="Directory to save results (default: exports)")
    
    parser.add_argument("--summary_length", choices=["short", "normal", "long"], default="normal",
                      help="Length of summary (default: normal)")
    
    parser.add_argument("--extract_keywords", action="store_true",
                      help="Extract keywords from document")
    
    return parser.parse_args()

def save_results(results, filename, output_dir):
    """Save processing results to files"""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    summary_path = os.path.join(output_dir, f"{base_name}.summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(results['summary'])
    
    # Save metrics if they exist
    if 'metrics' in results:
        metrics_path = os.path.join(output_dir, f"{base_name}.metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(results['metrics'], f, indent=2)
    
    # Save full results log
    log_path = os.path.join(output_dir, f"{base_name}.log")
    with open(log_path, 'w', encoding='utf-8') as f:
        # Copy results and truncate text to avoid huge log files
        log_results = results.copy()
        if 'text' in log_results and len(log_results['text']) > 1000:
            log_results['text'] = log_results['text'][:1000] + "... [truncated]"
        json.dump(log_results, f, indent=2)
    
    return {
        'summary_path': summary_path,
        'log_path': log_path,
        'metrics_path': metrics_path if 'metrics' in results else None
    }

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Setup output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    logger.info(f"Starting Smart Notes Summarizer CLI with mode: {args.mode}")
    logger.info(f"PDF path: {args.pdf_path}")
    logger.info(f"Summary length: {args.summary_length}")
    logger.info(f"Extract keywords: {args.extract_keywords}")
    
    try:
        # Initialize the processor
        processor = BatchPDFProcessor()
        
        if args.mode == "single":
            # Process a single PDF
            if not os.path.isfile(args.pdf_path):
                logger.error(f"File not found: {args.pdf_path}")
                sys.exit(1)
                
            logger.info(f"Processing single PDF: {args.pdf_path}")
            result = processor.process_pdf(
                args.pdf_path,
                summary_length=args.summary_length,
                extract_keywords=args.extract_keywords
            )
            
            # Save results
            output_files = save_results(result, args.pdf_path, args.output_dir)
            logger.info(f"Processing complete. Summary saved to {output_files['summary_path']}")
            
        elif args.mode == "batch":
            # Process all PDFs in a directory
            if not os.path.isdir(args.pdf_path):
                logger.error(f"Directory not found: {args.pdf_path}")
                sys.exit(1)
                
            logger.info(f"Processing all PDFs in directory: {args.pdf_path}")
            
            pdf_files = [os.path.join(args.pdf_path, f) for f in os.listdir(args.pdf_path) 
                         if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                logger.warning(f"No PDF files found in {args.pdf_path}")
                sys.exit(0)
                
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            for pdf_file in pdf_files:
                try:
                    logger.info(f"Processing {os.path.basename(pdf_file)}")
                    result = processor.process_pdf(
                        pdf_file,
                        summary_length=args.summary_length,
                        extract_keywords=args.extract_keywords
                    )
                    
                    # Save results
                    output_files = save_results(result, pdf_file, args.output_dir)
                    logger.info(f"Summary saved to {output_files['summary_path']}")
                    
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {str(e)}")
            
            logger.info("Batch processing complete")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()