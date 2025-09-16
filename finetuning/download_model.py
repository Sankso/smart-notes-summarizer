#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download and setup the pre-trained model and LoRA weights.
This script downloads the base model and sets up the directory structure
for the LoRA weights used in the Smart Notes Summarizer.
"""

import os
import logging
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

def download_model(model_name: str = "google/flan-t5-small", 
                  output_dir: str = "./models/lora_weights", 
                  download_base_only: bool = False):
    """
    Download model weights and create directories.
    
    Args:
        model_name: Base model name
        output_dir: Directory to save models
        create_lora_dir: Whether to create a directory for LoRA weights
        
    Returns:
        Dictionary with paths to model directories
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download base model
    logger.info(f"Downloading base model {model_name}")
    base_model_dir = os.path.join(output_dir, os.path.basename(model_name))
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(base_model_dir)
        
        # Download model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained(base_model_dir)
        
        logger.info(f"Base model downloaded to {base_model_dir}")
    except Exception as e:
        logger.error(f"Failed to download base model: {e}")
        raise
    
    # Create LoRA weights directory
    lora_dir = None
    if create_lora_dir:
        lora_dir = os.path.join(output_dir, "lora_weights")
        os.makedirs(lora_dir, exist_ok=True)
        
        # Create a placeholder README in the LoRA directory
        readme_path = os.path.join(lora_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write("# LoRA Weights Directory\n\n")
            f.write("This directory will contain LoRA adapter weights after fine-tuning.\n")
            f.write("To generate these weights, run:\n\n")
            f.write("```\n")
            f.write("python finetuning/train_lora.py \\\n")
            f.write("    --model_name google/flan-t5-small \\\n")
            f.write("    --dataset_path ./data/sample_dataset \\\n")
            f.write("    --output_dir ./models\n")
            f.write("```\n")
        
        logger.info(f"Created LoRA weights directory at {lora_dir}")
    
    return {
        "base_model_dir": base_model_dir,
        "lora_dir": lora_dir
    }


def main():
    """Main function to run model download from command line"""
    parser = argparse.ArgumentParser(description="Download model weights")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="google/flan-t5-small",
        help="Base model name"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./models",
        help="Directory to save models"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Download model
    download_model(args.model_name, args.output_dir)


if __name__ == "__main__":
    main()