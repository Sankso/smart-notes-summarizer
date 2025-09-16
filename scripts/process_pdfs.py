#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF Batch Processing Script for Smart Notes Summarizer.
This script processes PDF files using the fine-tuned model and generates summaries.
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import torch
import pdfplumber
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from rouge_score import rouge_scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_CHUNK_TOKENS = 512
MAX_TEXT_CHARS = 10000  # For very large PDFs
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lora_weights")
EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "exports")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def setup_directories() -> None:
    """
    Create necessary directories for exports and logs if they don't exist.
    """
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger.info(f"Setup directories: {EXPORTS_DIR}, {LOGS_DIR}")


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> Tuple[AutoTokenizer, Any]:
    """
    Load the fine-tuned model and tokenizer.
    
    Args:
        model_path: Path to the model directory
    
    Returns:
        Tuple of tokenizer and model
    """
    logger.info(f"Loading model from {model_path}")
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load base model and tokenizer
    base_model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    
    # Load LoRA weights if available
    if os.path.exists(model_path):
        try:
            model = PeftModel.from_pretrained(
                model,
                model_path,
                is_trainable=False
            )
            logger.info("LoRA weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LoRA weights: {e}")
            logger.warning("Falling back to base model")
    else:
        logger.warning(f"Model path {model_path} not found. Using base model.")
    
    # Move model to device
    model.to(device)
    
    return tokenizer, model


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    logger.info(f"Extracting text from {pdf_path}")
    extracted_text = ""
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n\n"
        
        logger.info(f"Extracted {len(extracted_text)} characters from PDF")
        return extracted_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return f"Error extracting text: {str(e)}"


def chunk_text(text: str, tokenizer: AutoTokenizer) -> List[str]:
    """
    Split text into chunks that won't exceed model's max token limit.
    
    Args:
        text: Text to split into chunks
        tokenizer: Tokenizer to count tokens
        
    Returns:
        List of text chunks
    """
    # First truncate if the text is extremely large
    if len(text) > MAX_TEXT_CHARS:
        logger.warning(f"Text too long ({len(text)} chars). Truncating to {MAX_TEXT_CHARS} chars.")
        text = text[:MAX_TEXT_CHARS]
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # Check if adding this paragraph would exceed the token limit
        temp_chunk = current_chunk + paragraph + "\n\n"
        tokens = tokenizer(temp_chunk, return_tensors="pt", truncation=False).input_ids
        
        if tokens.shape[1] > MAX_CHUNK_TOKENS and current_chunk:
            # If adding this paragraph exceeds limit and we have content, save current chunk
            chunks.append(current_chunk)
            current_chunk = paragraph + "\n\n"
        else:
            # Otherwise add paragraph to current chunk
            current_chunk = temp_chunk
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk)
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks


def summarize_chunk(chunk: str, tokenizer: AutoTokenizer, model: Any, summary_length: str = "normal") -> str:
    """
    Generate summary for a chunk of text using the fine-tuned model.
    
    Args:
        chunk: Text chunk to summarize
        tokenizer: Model tokenizer
        model: Fine-tuned model
        summary_length: Length of summary ("short", "normal", "long")
        
    Returns:
        Summary of the chunk
    """
    prompt = f"Summarize the following text:\n\n{chunk}"
    device = model.device
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_CHUNK_TOKENS).to(device)
    
    # Set generation parameters based on summary length
    max_new_tokens = 150
    min_new_tokens = 40
    
    if summary_length == "short":
        max_new_tokens = 75
        min_new_tokens = 30
    elif summary_length == "normal":
        max_new_tokens = 150
        min_new_tokens = 50
    elif summary_length == "long":
        max_new_tokens = 300
        min_new_tokens = 100
    
    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_length=min_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode and return summary
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def combine_chunk_summaries(chunk_summaries: List[str]) -> str:
    """
    Combine individual chunk summaries into a coherent final summary.
    
    Args:
        chunk_summaries: List of summaries from chunks
        
    Returns:
        Combined final summary
    """
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]
    
    # For multiple chunks, we need to further summarize
    combined_text = " ".join(chunk_summaries)
    return combined_text


def evaluate_summary(generated_summary: str, reference_summary: str) -> Dict[str, Any]:
    """
    Evaluate the quality of a generated summary using ROUGE metrics.
    
    Args:
        generated_summary: Model-generated summary
        reference_summary: Ground-truth/reference summary
        
    Returns:
        Dictionary with ROUGE scores
    """
    logger.info("Evaluating summary with ROUGE metrics")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    
    # Convert scores to a JSON-serializable format
    result = {
        'rouge1': {
            'precision': float(scores['rouge1'].precision),
            'recall': float(scores['rouge1'].recall),
            'fmeasure': float(scores['rouge1'].fmeasure)
        },
        'rouge2': {
            'precision': float(scores['rouge2'].precision),
            'recall': float(scores['rouge2'].recall),
            'fmeasure': float(scores['rouge2'].fmeasure)
        },
        'rougeL': {
            'precision': float(scores['rougeL'].precision),
            'recall': float(scores['rougeL'].recall),
            'fmeasure': float(scores['rougeL'].fmeasure)
        }
    }
    
    return result


def save_interaction_log(pdf_path: str, text: str, summary: str, metrics: Optional[Dict[str, Any]] = None) -> str:
    """
    Save interaction logs as a JSON file.
    
    Args:
        pdf_path: Path to the PDF file
        text: Extracted text content
        summary: Generated summary
        metrics: Evaluation metrics (optional)
        
    Returns:
        Path to the saved log file
    """
    filename = os.path.basename(pdf_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"{filename.split('.')[0]}_{timestamp}.json")
    
    # Create log data
    log_data = {
        "timestamp": timestamp,
        "pdf_filename": filename,
        "pdf_path": pdf_path,
        "text_snippet": text[:500] + "..." if len(text) > 500 else text,
        "text_length": len(text),
        "summary": summary,
        "summary_length": len(summary),
        "metrics": metrics or {}
    }
    
    # Save to JSON file
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Interaction log saved to: {log_path}")
    return log_path


def process_single_pdf(pdf_path: str, tokenizer: AutoTokenizer, model: Any, 
                      reference_summary: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a single PDF file: extract text, generate summary, evaluate.
    
    Args:
        pdf_path: Path to the PDF file
        tokenizer: Model tokenizer
        model: Fine-tuned model
        reference_summary: Optional reference summary for evaluation
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing PDF: {pdf_path}")
    start_time = time.time()
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    if not text or "Error extracting text" in text:
        result = {
            "success": False,
            "pdf_path": pdf_path,
            "error": text if "Error extracting text" in text else "No text extracted from PDF",
            "summary": "",
            "metrics": {}
        }
        return result
    
    # Split into chunks and summarize each
    chunks = chunk_text(text, tokenizer)
    logger.info(f"Summarizing {len(chunks)} chunks")
    
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
        summary = summarize_chunk(chunk, tokenizer, model)
        chunk_summaries.append(summary)
    
    # Combine chunk summaries for final summary
    final_summary = combine_chunk_summaries(chunk_summaries)
    
    # Evaluate if reference is provided
    metrics = {}
    if reference_summary:
        metrics = evaluate_summary(final_summary, reference_summary)
    
    # Save logs
    log_path = save_interaction_log(pdf_path, text, final_summary, metrics)
    
    # Prepare result
    process_time = time.time() - start_time
    result = {
        "success": True,
        "pdf_path": pdf_path,
        "summary": final_summary,
        "metrics": metrics,
        "log_path": log_path,
        "process_time_seconds": process_time
    }
    
    logger.info(f"PDF processed in {process_time:.2f} seconds")
    return result


def process_pdf_directory(pdf_dir: str, tokenizer: AutoTokenizer, model: Any) -> List[Dict[str, Any]]:
    """
    Process all PDF files in a directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        tokenizer: Model tokenizer
        model: Fine-tuned model
        
    Returns:
        List of processing results
    """
    logger.info(f"Processing all PDFs in directory: {pdf_dir}")
    
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    results = []
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files:
        result = process_single_pdf(pdf_file, tokenizer, model)
        results.append(result)
    
    return results


def main():
    """Main function to run the PDF processing script."""
    parser = argparse.ArgumentParser(description='Process PDF files with a fine-tuned summarization model')
    
    # Add arguments
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to PDF file or directory containing PDF files')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to fine-tuned model directory')
    parser.add_argument('--reference', type=str, default=None,
                        help='Path to reference summary for evaluation')
    
    args = parser.parse_args()
    
    # Set up directories
    setup_directories()
    
    # Load model
    tokenizer, model = load_model(args.model)
    
    # Process input (file or directory)
    if os.path.isdir(args.input):
        results = process_pdf_directory(args.input, tokenizer, model)
        logger.info(f"Processed {len(results)} PDF files. Results saved to {LOGS_DIR}")
    else:
        reference_summary = None
        if args.reference and os.path.exists(args.reference):
            with open(args.reference, 'r', encoding='utf-8') as f:
                reference_summary = f.read()
        
        result = process_single_pdf(args.input, tokenizer, model, reference_summary)
        if result["success"]:
            logger.info(f"Summary: {result['summary']}")
            logger.info(f"Log saved to: {result['log_path']}")
        else:
            logger.error(f"Error: {result['error']}")


if __name__ == "__main__":
    main()