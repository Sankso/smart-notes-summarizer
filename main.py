#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Smart Notes Summarizer Agent - Main entry point
"""

import argparse
import sys
import logging
from agent.agent import SmartSummarizerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(
        description='Smart Notes Summarizer Agent - Summarize PDFs and lecture notes'
    )
    
    parser.add_argument(
        '--pdf', 
        type=str,
        help='Path to PDF file to summarize'
    )
    
    parser.add_argument(
        '--text', 
        type=str,
        help='Text content to summarize'
    )
    
    parser.add_argument(
        '--output', 
        type=str,
        default='summary.txt',
        help='Output file to save summary (default: summary.txt)'
    )
    
    parser.add_argument(
        '--ui', 
        action='store_true',
        help='Launch the Streamlit UI'
    )
    
    args = parser.parse_args()
    
    # Launch UI if requested
    if args.ui:
        logger.info("Launching Streamlit UI...")
        import subprocess
        import os
        subprocess.run(['streamlit', 'run', 
                       os.path.join('ui', 'app.py')])
        return
    
    # Ensure we have either PDF or text input
    if not args.pdf and not args.text:
        parser.error("You must provide either --pdf or --text")
    
    # Initialize agent
    agent = SmartSummarizerAgent()
    
    # Process input and get summary
    if args.pdf:
        logger.info(f"Summarizing PDF: {args.pdf}")
        summary = agent.summarize_pdf(args.pdf)
    else:
        logger.info("Summarizing provided text")
        summary = agent.summarize_text(args.text)
    
    # Print summary to console
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    print(summary)
    print("="*50 + "\n")
    
    # Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    logger.info(f"Summary saved to {args.output}")

if __name__ == "__main__":
    main()