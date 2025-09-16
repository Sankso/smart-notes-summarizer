#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for summarization models.
Runs inference on test datasets and calculates quality metrics.
"""

import os
import json
import logging
import argparse
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from datasets import load_from_disk, Dataset
from tqdm import tqdm

from evaluation.metrics import SummarizationMetrics
from finetuning.inference import SummarizerInference

logger = logging.getLogger(__name__)

class SummarizerEvaluator:
    """
    Evaluates summarization models on test datasets.
    Runs inference and calculates ROUGE and BLEU metrics.
    """
    
    def __init__(self,
                model_name: str = "google/flan-t5-small",
                lora_weights_dir: Optional[str] = None,
                output_dir: str = "./evaluation_results",
                device: str = None):
        """
        Initialize the evaluator.
        
        Args:
            model_name: Base model name
            lora_weights_dir: Directory containing LoRA adapter weights
            output_dir: Directory to save evaluation results
            device: Device to run evaluation on ('cuda', 'cpu')
        """
        self.model_name = model_name
        self.lora_weights_dir = lora_weights_dir
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics calculator
        self.metrics = SummarizationMetrics()
        
        # Initialize inference module
        self.inference = SummarizerInference(
            model_name=model_name,
            lora_weights_dir=lora_weights_dir,
            device=device
        )
        
        logger.info(f"Evaluator initialized with {model_name}")
        if lora_weights_dir:
            logger.info(f"Using LoRA weights from {lora_weights_dir}")
    
    def evaluate_dataset(self,
                        dataset_path: str,
                        split: str = "test",
                        input_column: str = "text",
                        reference_column: str = "summary",
                        max_samples: Optional[int] = None,
                        save_predictions: bool = True) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset_path: Path to the dataset
            split: Dataset split to evaluate on
            input_column: Column containing input text
            reference_column: Column containing reference summaries
            max_samples: Maximum number of samples to evaluate
            save_predictions: Whether to save predictions to a file
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating model on dataset: {dataset_path}, split: {split}")
        
        # Load dataset
        try:
            dataset = load_from_disk(dataset_path)
            test_data = dataset[split]
            logger.info(f"Loaded {len(test_data)} examples from {split} split")
        except Exception as e:
            logger.error(f"Failed to load dataset from {dataset_path}: {e}")
            raise
        
        # Limit number of samples if specified
        if max_samples and max_samples < len(test_data):
            test_data = test_data.select(range(max_samples))
            logger.info(f"Limited evaluation to {max_samples} examples")
        
        # Generate predictions
        predictions = []
        references = []
        inputs = []
        
        logger.info("Generating summaries...")
        for example in tqdm(test_data):
            # Get input and reference
            input_text = example[input_column]
            reference = example[reference_column]
            
            # Generate summary
            result = self.inference.summarize(input_text)
            prediction = result["summary"]
            
            # Store results
            inputs.append(input_text)
            predictions.append(prediction)
            references.append(reference)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = self.metrics.calculate_batch_metrics(predictions, references)
        
        # Create evaluation result
        result = {
            "model": self.model_name,
            "lora_weights": self.lora_weights_dir,
            "dataset": dataset_path,
            "split": split,
            "num_examples": len(predictions),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"evaluation_{os.path.basename(dataset_path)}_{timestamp}.json"
        results_path = os.path.join(self.output_dir, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
        
        # Save predictions if requested
        if save_predictions:
            predictions_df = pd.DataFrame({
                "input": inputs,
                "reference": references,
                "prediction": predictions
            })
            
            predictions_filename = f"predictions_{os.path.basename(dataset_path)}_{timestamp}.csv"
            predictions_path = os.path.join(self.output_dir, predictions_filename)
            predictions_df.to_csv(predictions_path, index=False)
            
            logger.info(f"Predictions saved to {predictions_path}")
        
        return result
    
    def evaluate_examples(self,
                        inputs: List[str],
                        references: List[str]) -> Dict[str, Any]:
        """
        Evaluate the model on a list of examples.
        
        Args:
            inputs: List of input texts
            references: List of reference summaries
            
        Returns:
            Dictionary of evaluation results
        """
        if len(inputs) != len(references):
            raise ValueError("Number of inputs and references must match")
        
        logger.info(f"Evaluating model on {len(inputs)} examples")
        
        # Generate predictions
        predictions = []
        for input_text in tqdm(inputs):
            # Generate summary
            result = self.inference.summarize(input_text)
            prediction = result["summary"]
            predictions.append(prediction)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = self.metrics.calculate_batch_metrics(predictions, references)
        
        # Create evaluation result
        result = {
            "model": self.model_name,
            "lora_weights": self.lora_weights_dir,
            "num_examples": len(predictions),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics
        }
        
        return result
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """
        Format evaluation results as a human-readable string.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Formatted results string
        """
        output = []
        
        # Add header
        output.append("# Summarization Evaluation Results\n")
        
        # Model information
        output.append(f"- **Model**: {results['model']}")
        if results.get('lora_weights'):
            output.append(f"- **LoRA weights**: {results['lora_weights']}")
        output.append(f"- **Dataset**: {results.get('dataset', 'custom examples')}")
        output.append(f"- **Examples**: {results['num_examples']}")
        output.append(f"- **Date**: {results['timestamp']}\n")
        
        # ROUGE scores
        output.append("## ROUGE Scores\n")
        output.append("| Metric | Precision | Recall | F1 |")
        output.append("|--------|-----------|--------|----|\n")
        
        for rouge_type in ["rouge1", "rouge2", "rougeL"]:
            precision = results["metrics"]["rouge"][rouge_type]["precision"]["mean"]
            recall = results["metrics"]["rouge"][rouge_type]["recall"]["mean"]
            f1 = results["metrics"]["rouge"][rouge_type]["fmeasure"]["mean"]
            
            output.append(f"| {rouge_type} | {precision:.4f} | {recall:.4f} | {f1:.4f} |")
        
        # BLEU scores
        output.append("\n## BLEU Scores\n")
        output.append("| Metric | Score |")
        output.append("|--------|-------|\n")
        
        for bleu_type in ["bleu1", "bleu2", "bleu4"]:
            score = results["metrics"]["bleu"][bleu_type]["mean"]
            output.append(f"| {bleu_type} | {score:.4f} |")
        
        return "\n".join(output)


def main():
    """Main function to run evaluation from command line"""
    parser = argparse.ArgumentParser(description="Evaluate summarization models")
    
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
        "--dataset", 
        type=str, 
        required=True,
        help="Path to the dataset"
    )
    
    parser.add_argument(
        "--split", 
        type=str, 
        default="test",
        help="Dataset split to evaluate on"
    )
    
    parser.add_argument(
        "--input_column", 
        type=str, 
        default="text",
        help="Column containing input text"
    )
    
    parser.add_argument(
        "--reference_column", 
        type=str, 
        default="summary",
        help="Column containing reference summaries"
    )
    
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to evaluate"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to run evaluation on ('cpu', 'cuda')"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Initialize evaluator
    evaluator = SummarizerEvaluator(
        model_name=args.model_name,
        lora_weights_dir=args.lora_weights,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        dataset_path=args.dataset,
        split=args.split,
        input_column=args.input_column,
        reference_column=args.reference_column,
        max_samples=args.max_samples
    )
    
    # Format and print results
    formatted_results = evaluator.format_results(results)
    print("\n" + formatted_results)


if __name__ == "__main__":
    main()