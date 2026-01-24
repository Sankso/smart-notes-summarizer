#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics calculation module for summarization evaluation.
Provides standard ROUGE and BLEU metric calculations.
"""

import logging
import nltk
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

class SummarizationMetrics:
    """
    Calculates ROUGE and BLEU metrics for summarization tasks.
    """
    
    def __init__(self, use_stemmer=True):
        """
        Initialize the metrics calculator.
        
        Args:
            use_stemmer: Whether to use stemming for ROUGE calculation
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK resources: {e}")
                
    def calculate_rouge(self, prediction, reference):
        """
        Calculate ROUGE scores for a single prediction-reference pair.
        """
        return self.rouge_scorer.score(reference, prediction)
    
    def calculate_bleu(self, prediction, reference):
        """
        Calculate BLEU-like scores based on n-gram overlap for a single pair.
        """
        # Tokenize text
        reference_tokens = reference.lower().split()
        prediction_tokens = prediction.lower().split()
        
        # Create n-grams
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        # Unigram overlap (BLEU-1 like)
        ref_unigrams = set(reference_tokens)
        pred_unigrams = set(prediction_tokens)
        overlap_unigrams = ref_unigrams.intersection(pred_unigrams)
        bleu1 = len(overlap_unigrams) / max(1, len(pred_unigrams)) if pred_unigrams else 0
        
        # Bigram overlap (BLEU-2 like)
        ref_bigrams = set(get_ngrams(reference_tokens, 2))
        pred_bigrams = set(get_ngrams(prediction_tokens, 2))
        overlap_bigrams = ref_bigrams.intersection(pred_bigrams)
        bleu2 = len(overlap_bigrams) / max(1, len(pred_bigrams)) if pred_bigrams else 0
        
        # 4-gram overlap (BLEU-4 like)
        ref_4grams = set(get_ngrams(reference_tokens, 4))
        pred_4grams = set(get_ngrams(prediction_tokens, 4))
        overlap_4grams = ref_4grams.intersection(pred_4grams)
        bleu4 = len(overlap_4grams) / max(1, len(pred_4grams)) if pred_4grams else 0
        
        return {
            "bleu1": bleu1,
            "bleu2": bleu2,
            "bleu4": bleu4
        }
    
    def calculate_single_metrics(self, prediction, reference):
        """
        Calculate all metrics for a single prediction-reference pair.
        """
        rouge_scores = self.calculate_rouge(prediction, reference)
        bleu_scores = self.calculate_bleu(prediction, reference)
        
        # Calculate lexical diversity
        prediction_tokens = prediction.lower().split()
        lexical_diversity = len(set(prediction_tokens)) / len(prediction_tokens) if prediction_tokens else 0
        
        # Calculate other summary statistics
        summary_stats = {
            "length": len(prediction),
            "num_sentences": len(nltk.sent_tokenize(prediction)),
            "lexical_diversity": lexical_diversity
        }
        
        return {
            "rouge": rouge_scores,
            "bleu": bleu_scores,
            
            # Flattened ROUGE F1s for easy access
            "rouge1_f1": rouge_scores["rouge1"].fmeasure,
            "rouge2_f1": rouge_scores["rouge2"].fmeasure,
            "rougeL_f1": rouge_scores["rougeL"].fmeasure,
            
            # Flattened BLEU for easy access
            "bleu1": bleu_scores["bleu1"],
            "bleu2": bleu_scores["bleu2"],
            "bleu4": bleu_scores["bleu4"],
            
            "stats": summary_stats
        }

    def calculate_batch_metrics(self, predictions, references):
        """
        Calculate average metrics for a batch of predictions and references.
        Returns a dictionary structure compatible with the original evaluate.py expectations.
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
            
        rouge_totals = {
            "rouge1": {"precision": [], "recall": [], "fmeasure": []},
            "rouge2": {"precision": [], "recall": [], "fmeasure": []},
            "rougeL": {"precision": [], "recall": [], "fmeasure": []}
        }
        
        bleu_totals = {
            "bleu1": [],
            "bleu2": [],
            "bleu4": []
        }
        
        for pred, ref in zip(predictions, references):
            scores = self.calculate_single_metrics(pred, ref)
            
            # Aggregate ROUGE
            for r_type in ["rouge1", "rouge2", "rougeL"]:
                rouge_totals[r_type]["precision"].append(scores["rouge"][r_type].precision)
                rouge_totals[r_type]["recall"].append(scores["rouge"][r_type].recall)
                rouge_totals[r_type]["fmeasure"].append(scores["rouge"][r_type].fmeasure)
                
            # Aggregate BLEU
            for b_type in ["bleu1", "bleu2", "bleu4"]:
                bleu_totals[b_type].append(scores["bleu"][b_type])
                
        # Calculate means
        results = {
            "rouge": {},
            "bleu": {}
        }
        
        from statistics import mean
        
        for r_type in rouge_totals:
            results["rouge"][r_type] = {
                "precision": {"mean": mean(rouge_totals[r_type]["precision"])},
                "recall": {"mean": mean(rouge_totals[r_type]["recall"])},
                "fmeasure": {"mean": mean(rouge_totals[r_type]["fmeasure"])}
            }
            
        for b_type in bleu_totals:
            results["bleu"][b_type] = {"mean": mean(bleu_totals[b_type]) if bleu_totals[b_type] else 0}
            
        return results
