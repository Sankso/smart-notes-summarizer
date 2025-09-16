#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to check model loading
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def test_model_loading():
    """Test if the model can be loaded correctly"""
    try:
        logger.info("Testing imports...")
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        
        logger.info("Imports successful")
        
        # Test loading base model
        model_name = "google/flan-t5-small"
        logger.info(f"Loading base model {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        logger.info("Base model loaded successfully")
        
        # Test LoRA
        try:
            from peft import PeftModel, PeftConfig
            logger.info("PEFT imported successfully")
        except ImportError:
            logger.error("Failed to import PEFT. Make sure it's installed")
            return
        
        # Try to load LoRA weights
        lora_weights_dir = os.path.join(parent_dir, "smart-notes-summarizer", "models", "lora_weights")
        if not os.path.exists(lora_weights_dir):
            # Try alternative path
            lora_weights_dir = os.path.join(parent_dir, "models", "lora_weights")
        
        if os.path.exists(lora_weights_dir):
            logger.info(f"Testing LoRA weights loading from {lora_weights_dir}")
            
            # Check adapter config
            adapter_config_path = os.path.join(lora_weights_dir, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                logger.info(f"adapter_config.json exists at {adapter_config_path}")
                
                # Try loading the config
                try:
                    import json
                    with open(adapter_config_path, 'r') as f:
                        config_data = json.load(f)
                        logger.info(f"Config keys: {list(config_data.keys())}")
                        
                        # Check for problematic keys
                        problematic_keys = ["corda_config", "eva_config"]
                        for key in problematic_keys:
                            if key in config_data:
                                logger.warning(f"Found potentially problematic key in config: {key}")
                except Exception as e:
                    logger.error(f"Error reading adapter_config.json: {e}")
            
            # Try loading the model with LoRA
            try:
                model = PeftModel.from_pretrained(
                    model,
                    lora_weights_dir,
                    is_trainable=False
                )
                logger.info("LoRA weights loaded successfully!")
                
                # Try creating a pipeline
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1  # Use CPU
                )
                logger.info("Pipeline created successfully")
                
                # Test generation
                test_input = "Summarize this text: This is a test of the model loading."
                logger.info(f"Testing generation with input: {test_input}")
                result = pipe(test_input, max_length=50)
                logger.info(f"Generation result: {result}")
                
            except Exception as e:
                logger.error(f"Failed to load LoRA weights: {e}")
                logger.error("Falling back to base model")
                
                # Try with base model
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1  # Use CPU
                )
                logger.info("Pipeline with base model created successfully")
                
                test_input = "Summarize this text: This is a test of the model loading."
                logger.info(f"Testing generation with input: {test_input}")
                result = pipe(test_input, max_length=50)
                logger.info(f"Generation result: {result}")
        else:
            logger.warning(f"LoRA weights directory not found at {lora_weights_dir}")
    
    except Exception as e:
        logger.error(f"Error testing model loading: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Starting model loading test")
    test_model_loading()
    logger.info("Test completed")