#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning script with Parameter-Efficient Fine-Tuning (LoRA) for summarization models.
Based on the successful notebook used to create the finetuned model.
"""

import os
import argparse
import logging
import torch
from pathlib import Path
import shutil

from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq,
    TrainingArguments, 
    Trainer
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)
    """
    LoRA fine-tuning for seq2seq models (e.g., T5).
    Uses Parameter-Efficient Fine-Tuning to train only a small set of parameters.
    """
    
    def __init__(self,
                model_name: str = "google/flan-t5-small",
                output_dir: str = "./models",
                lora_r: int = 16,
                lora_alpha: int = 32,
                lora_dropout: float = 0.1,
                max_input_length: int = 512,
                max_output_length: int = 150):
        """
        Initialize the LoRA trainer.
        
        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save fine-tuned model
            lora_r: LoRA attention dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initializing LoRA trainer with {model_name}")
        logger.info(f"LoRA parameters: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    def train(self,
             dataset_path: str,
             per_device_train_batch_size: int = 8,
             gradient_accumulation_steps: int = 1,
             learning_rate: float = 5e-4,
             num_train_epochs: int = 3,
             warmup_ratio: float = 0.1,
             weight_decay: float = 0.01,
             fp16: bool = False,
             bf16: bool = False,
             target_modules: Optional[List[str]] = None,
             eval_steps: int = 500,
             save_steps: int = 1000) -> str:
        """
        Train the model using LoRA.
        
        Args:
            dataset_path: Path to the processed dataset
            per_device_train_batch_size: Batch size per GPU/TPU core
            gradient_accumulation_steps: Number of updates steps to accumulate before backward pass
            learning_rate: Initial learning rate
            num_train_epochs: Number of training epochs
            warmup_ratio: Ratio of steps for learning rate warmup
            weight_decay: Weight decay rate
            fp16: Whether to use 16-bit (mixed) precision
            bf16: Whether to use bf16 (mixed) precision
            target_modules: List of modules to apply LoRA to (if None, use default)
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between checkpoints
            
        Returns:
            Path to the fine-tuned model
        """
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load the dataset
        try:
            dataset = load_from_disk(dataset_path)
            logger.info(f"Dataset loaded with splits: {', '.join(dataset.keys())}")
        except Exception as e:
            logger.error(f"Failed to load dataset from {dataset_path}: {e}")
            raise
        
        # Load tokenizer
        logger.info(f"Loading tokenizer for {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Prepare dataset
        logger.info("Preparing dataset for training")
        tokenized_dataset = self._prepare_dataset(dataset, tokenizer)
        
        # Load base model
        logger.info(f"Loading base model {self.model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Define LoRA configuration
        if target_modules is None:
            # Default target modules for T5 models
            target_modules = ["q", "v"]
        
        logger.info(f"Configuring LoRA for target modules: {target_modules}")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
        )
        
        # Wrap model with LoRA
        logger.info("Applying LoRA adapters to model")
        model = get_peft_model(model, lora_config)
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Parameter efficiency: {trainable_params/total_params*100:.2f}%")
        
        # Prepare data collator
        logger.info("Preparing data collator")
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="longest"
        )
        
        # Create a timestamp for the run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"lora_{os.path.basename(self.model_name)}_{timestamp}"
        
        # Define training arguments
        logger.info("Setting up training arguments")
        training_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(self.output_dir, run_name),
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            fp16=fp16,
            bf16=bf16,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            logging_dir=os.path.join(self.output_dir, run_name, "logs"),
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["tensorboard"],
            remove_unused_columns=True
        )
        
        # Create trainer
        logger.info("Creating trainer")
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("validation", None),
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        
        # Train the model
        logger.info("Starting training")
        trainer.train()
        
        # Save the final model
        final_model_path = os.path.join(self.output_dir, "lora_weights")
        logger.info(f"Saving LoRA weights to {final_model_path}")
        
        # Save only LoRA weights
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        logger.info("Training completed successfully")
        return final_model_path
    
    def _prepare_dataset(self, dataset: DatasetDict, tokenizer) -> DatasetDict:
        """
        Prepare and tokenize the dataset.
        
        Args:
            dataset: Input dataset
            tokenizer: Tokenizer to use
            
        Returns:
            Tokenized dataset
        """
        # Add prefix for T5 models
        prefix = "summarize: "
        
        def preprocess_function(examples):
            # Add prefix to the inputs
            inputs = [prefix + doc for doc in examples["text"]]
            
            # Tokenize inputs
            model_inputs = tokenizer(
                inputs, 
                max_length=self.max_input_length,
                truncation=True,
                padding=False
            )
            
            # Tokenize targets
            labels = tokenizer(
                examples["summary"],
                max_length=self.max_output_length,
                truncation=True,
                padding=False
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Apply preprocessing to each split
        tokenized_dataset = {}
        for split, data in dataset.items():
            tokenized_dataset[split] = data.map(
                preprocess_function,
                batched=True,
                desc=f"Tokenizing {split} split"
            )
            
            # Log dataset stats
            logger.info(f"{split} split: {len(tokenized_dataset[split])} examples")
        
        return DatasetDict(tokenized_dataset)


def main():
    """Main function to run LoRA fine-tuning from command line"""
    parser = argparse.ArgumentParser(description="Fine-tune models with LoRA for summarization")
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="google/flan-t5-small",
        help="Base model to fine-tune"
    )
    
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True,
        help="Path to the processed dataset"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./models",
        help="Directory to save fine-tuned model"
    )
    
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=16,
        help="LoRA attention dimension"
    )
    
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=32,
        help="LoRA alpha parameter"
    )
    
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.1,
        help="Dropout probability for LoRA layers"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Training batch size per device"
    )
    
    parser.add_argument(
        "--gradient_accumulation", 
        type=int, 
        default=1,
        help="Number of updates steps to accumulate before backward pass"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-4,
        help="Initial learning rate"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--max_input_length", 
        type=int, 
        default=512,
        help="Maximum input sequence length"
    )
    
    parser.add_argument(
        "--max_output_length", 
        type=int, 
        default=150,
        help="Maximum output sequence length"
    )
    
    parser.add_argument(
        "--fp16", 
        action="store_true",
        help="Whether to use 16-bit (mixed) precision"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Initialize trainer
    trainer = LoRATrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length
    )
    
    # Run training
    trainer.train(
        dataset_path=args.dataset_path,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        fp16=args.fp16
    )


if __name__ == "__main__":
    main()