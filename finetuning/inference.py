#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference module for fine-tuned summarization models with LoRA.
Handles loading of models with LoRA weights and generation of summaries.
This script is designed to work with the model fine-tuned in train_lora_notebook.ipynb.
"""

import os
import time
import logging
import argparse
import torch
from typing import Dict, List, Optional, Union, Any

from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer
)
from peft import PeftModel, PeftConfig
from peft import PeftModel, PeftConfig

logger = logging.getLogger(__name__)

class SummarizerInference:
    """
    Handles inference for fine-tuned summarization models.
    Supports loading base models with LoRA adapters for efficient inference.
    """
    
    def __init__(self,
                model_name: str = "google/flan-t5-small",
                lora_weights_dir: Optional[str] = None,
                device: str = None,
                max_length: int = 150,
                min_length: int = 30):
        """
        Initialize the inference module.
        
        Args:
            model_name: Base model name
            lora_weights_dir: Directory containing LoRA adapter weights
            device: Device to run model on ('cuda', 'cpu', etc.)
            max_length: Maximum output length
            min_length: Minimum output length
        """
        # Set device (use CUDA if available)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_name = model_name
        self.lora_weights_dir = lora_weights_dir
        self.max_length = max_length
        self.min_length = min_length
        
        # Load model and tokenizer
        self._load_model()
        
        logger.info(f"Inference module initialized with {model_name} on {self.device}")
        if lora_weights_dir:
            logger.info(f"Using LoRA weights from {lora_weights_dir}")
    
    def _load_model(self):
        """Load the model and tokenizer with LoRA weights if available"""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load base model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Apply LoRA weights if provided
        if self.lora_weights_dir and os.path.exists(self.lora_weights_dir):
            logger.info(f"Loading LoRA weights from {self.lora_weights_dir}")
            try:
                # Handle adapter config with compatibility fixes
                import json
                
                # Load and modify config if needed
                config_path = os.path.join(self.lora_weights_dir, "adapter_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    # Remove problematic parameters if they exist
                    if 'corda_config' in config_data:
                        logger.info("Removing 'corda_config' parameter for compatibility")
                        del config_data['corda_config']
                    
                    # Write back modified config
                    with open(config_path, 'w') as f:
                        json.dump(config_data, f, indent=2)
                
                # Now load the modified configuration
                from peft import PeftConfig
                config = PeftConfig.from_pretrained(self.lora_weights_dir)
                
                # Load base model and LoRA adapter
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_weights_dir,
                    is_trainable=False
                )
                logger.info("LoRA weights loaded successfully")
                
            except Exception as e:
                logger.warning(f"Failed to load LoRA weights: {e}")
                logger.warning(f"Error details: {str(e)}")
                logger.warning("Falling back to base model")
        
        # Move model to device
        self.model.to(self.device)
        
        # Create generation pipeline
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            max_length=self.max_length,
            min_length=self.min_length
        )
    
    def download_model(output_dir: str = "./models/lora_weights"):
        """
        Download the fine-tuned LoRA weights.
        
        Args:
            output_dir: Directory to save the model
            
        Returns:
            Path to the downloaded model
        """
        logger.info(f"This function would download pre-trained weights to {output_dir}")
        logger.info("For this project, please fine-tune your own model or use the base model")
        
        # In a real project, you would download weights from a server
        # For example:
        # import requests
        # url = "https://your-model-storage/lora_weights.zip"
        # r = requests.get(url)
        # with open("lora_weights.zip", "wb") as f:
        #     f.write(r.content)
        # Then unzip the file...
        
        return output_dir
    
    def summarize(self, text: str, **generation_kwargs) -> Dict[str, Any]:
        """
        Generate a summary for the input text.
        
        Args:
            text: Input text to summarize
            generation_kwargs: Additional keyword arguments for text generation
            
        Returns:
            Dictionary containing summary and metadata
        """
        start_time = time.time()
        
        # Add prefix for T5 models
        prompt = f"summarize: {text}"
        
        # Set default generation parameters
        params = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "no_repeat_ngram_size": 3
        }
        
        # Update with any user-provided parameters
        params.update(generation_kwargs)
        
        # Generate summary
        logger.info(f"Generating summary for text ({len(text)} chars)")
        result = self.pipe(
            prompt,
            **params
        )
        
        summary = result[0]['generated_text']
        generation_time = time.time() - start_time
        
        logger.info(f"Summary generated ({len(summary)} chars) in {generation_time:.2f}s")
        
        return {
            "summary": summary,
            "input_length": len(text),
            "output_length": len(summary),
            "generation_time": generation_time
        }
    
    def batch_summarize(self, texts: List[str], **generation_kwargs) -> List[Dict[str, Any]]:
        """
        Generate summaries for a batch of texts.
        
        Args:
            texts: List of input texts to summarize
            generation_kwargs: Additional keyword arguments for text generation
            
        Returns:
            List of dictionaries containing summaries and metadata
        """
        start_time = time.time()
        
        # Add prefix for T5 models
        prompts = [f"summarize: {text}" for text in texts]
        
        # Set default generation parameters
        params = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "no_repeat_ngram_size": 3
        }
        
        # Update with any user-provided parameters
        params.update(generation_kwargs)
        
        # Generate summaries
        logger.info(f"Generating summaries for {len(texts)} texts")
        results = self.pipe(
            prompts,
            **params
        )
        
        # Process results
        summaries = []
        for i, result in enumerate(results):
            summary = result['generated_text']
            summaries.append({
                "summary": summary,
                "input_length": len(texts[i]),
                "output_length": len(summary)
            })
        
        total_time = time.time() - start_time
        logger.info(f"Generated {len(texts)} summaries in {total_time:.2f}s")
        
        return summaries


def main():
    """Main function to run inference from command line"""
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned summarization model")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="google/flan-t5-small",
        help="Base model name"
    )
    
    parser.add_argument(
        "--lora_weights", 
        type=str, 
        default=None,
        help="Path to LoRA weights directory"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to run inference on ('cpu', 'cuda')"
    )
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        default=None,
        help="Path to input text file"
    )
    
    parser.add_argument(
        "--text", 
        type=str, 
        default=None,
        help="Text to summarize"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="Path to save the summary output"
    )
    
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=150,
        help="Maximum output length"
    )
    
    parser.add_argument(
        "--min_length", 
        type=int, 
        default=30,
        help="Minimum output length"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Initialize inference module
    summarizer = SummarizerInference(
        model_name=args.model_name,
        lora_weights_dir=args.lora_weights,
        device=args.device,
        max_length=args.max_length,
        min_length=args.min_length
    )
    
    # Get input text
    if args.text:
        text = args.text
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = input("Enter text to summarize: ")
    
    # Generate summary
    result = summarizer.summarize(text)
    summary = result['summary']
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    print(summary)
    print("="*50 + "\n")
    
    # Save to file if specified
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"Summary saved to {args.output_file}")


if __name__ == "__main__":
    main()