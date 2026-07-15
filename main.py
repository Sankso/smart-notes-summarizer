#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Smart Notes Summarizer — CLI Entry Point

Usage:
    python main.py --pdf <path>          Summarize a PDF document
    python main.py --text "Your text"    Summarize text directly
    python main.py --pdf <path> --output results.txt
"""

import argparse
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from agent.agent import SmartSummarizerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='Smart Notes Summarizer — Summarize PDFs and lecture notes'
    )
    
    parser.add_argument('--pdf', type=str, help='Path to PDF file to summarize')
    parser.add_argument('--text', type=str, help='Text content to summarize')
    parser.add_argument('--length', type=str, default='normal',
                       choices=['short', 'normal', 'long'],
                       help='Summary length (default: normal)')
    parser.add_argument('--output', type=str, default='summary.txt',
                       help='Output file to save summary (default: summary.txt)')
    
    args = parser.parse_args()
    
    # Ensure we have either PDF or text input
    if not args.pdf and not args.text:
        parser.error("You must provide either --pdf or --text")
    
    # Initialize agent
    agent = SmartSummarizerAgent()
    
    # Process input
    if args.pdf:
        logger.info(f"Summarizing PDF: {args.pdf}")
        result = agent.summarize_pdf(args.pdf, summary_length=args.length)
    else:
        logger.info("Summarizing provided text")
        result = agent.summarize_text(args.text, summary_length=args.length)
    
    # Display results
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(result.get("summary", ""))
    
    if result.get("keywords"):
        print("\nKEYWORDS:")
        keywords = [kw['keyword'] for kw in result['keywords']]
        print(", ".join(keywords))
    
    if result.get("stats"):
        stats = result["stats"]
        print(f"\nPIPELINE STATS:")
        print(f"  Chunks: {stats.get('num_chunks', 1)}")
        print(f"  Routed to FLAN-T5: {stats.get('chunks_local', 0)}")
        print(f"  Routed to Gemini:  {stats.get('chunks_gemini', 0)}")
    
    if result.get("chunk_details"):
        print(f"\nPER-CHUNK ROUTING:")
        for detail in result["chunk_details"]:
            kws = ", ".join(detail.get("keywords", [])[:3])
            print(f"  Chunk {detail['chunk_index']+1}: {detail['routing']} "
                  f"| keywords: [{kws}]")
    
    print("=" * 60 + "\n")
    
    # Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(result.get("summary", ""))
    
    logger.info(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()