#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the evaluation metrics.
Tests ROUGE and BLEU score calculations.
"""

import os
import sys
import unittest
import logging
import numpy as np
from unittest import mock

# Add parent directory to path to import project modules
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from evaluation.metrics import SummarizationMetrics

# Disable logging during tests
logging.disable(logging.CRITICAL)

class TestSummarizationMetrics(unittest.TestCase):
    """Test cases for the Summarization Metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.metrics = SummarizationMetrics(use_stemmer=True)
        
        # Sample texts for testing
        self.reference = "The quick brown fox jumps over the lazy dog. It was a sunny day in the forest."
        self.good_prediction = "The quick brown fox jumps over the lazy dog. The day was sunny in the forest."
        self.bad_prediction = "A fox jumped. It was sunny."
        self.unrelated_prediction = "The weather was cold and rainy in the city."
    
    def test_rouge_scores_good_prediction(self):
        """Test ROUGE scores with a good prediction"""
        scores = self.metrics.calculate_rouge(self.good_prediction, self.reference)
        
        # Check that all expected ROUGE scores are present
        self.assertIn('rouge1', scores)
        self.assertIn('rouge2', scores)
        self.assertIn('rougeL', scores)
        
        # Check structure of each score
        for rouge_type, score in scores.items():
            self.assertIn('precision', score)
            self.assertIn('recall', score)
            self.assertIn('fmeasure', score)
        
        # Good prediction should have high ROUGE-1 F1 score
        self.assertGreater(scores['rouge1']['fmeasure'], 0.7)
    
    def test_rouge_scores_bad_prediction(self):
        """Test ROUGE scores with a bad prediction"""
        scores = self.metrics.calculate_rouge(self.bad_prediction, self.reference)
        
        # Bad prediction should have lower ROUGE-1 F1 score
        self.assertLess(scores['rouge1']['fmeasure'], 0.7)
        
        # Bad prediction should have very low ROUGE-2 score due to missing bigrams
        self.assertLess(scores['rouge2']['fmeasure'], 0.5)
    
    def test_rouge_scores_unrelated_prediction(self):
        """Test ROUGE scores with an unrelated prediction"""
        scores = self.metrics.calculate_rouge(self.unrelated_prediction, self.reference)
        
        # Unrelated prediction should have very low scores
        self.assertLess(scores['rouge1']['fmeasure'], 0.3)
        self.assertLess(scores['rouge2']['fmeasure'], 0.1)
    
    def test_bleu_scores_good_prediction(self):
        """Test BLEU scores with a good prediction"""
        scores = self.metrics.calculate_bleu(self.good_prediction, self.reference)
        
        # Check that all expected BLEU scores are present
        self.assertIn('bleu1', scores)
        self.assertIn('bleu2', scores)
        self.assertIn('bleu4', scores)
        
        # Good prediction should have reasonable BLEU-1 score
        self.assertGreater(scores['bleu1'], 0.6)
    
    def test_bleu_scores_bad_prediction(self):
        """Test BLEU scores with a bad prediction"""
        scores = self.metrics.calculate_bleu(self.bad_prediction, self.reference)
        
        # Bad prediction should have lower BLEU scores
        self.assertLess(scores['bleu4'], 0.5)
    
    def test_all_metrics(self):
        """Test combined metrics calculation"""
        metrics = self.metrics.calculate_metrics(self.good_prediction, self.reference)
        
        # Check that all expected metric groups are present
        self.assertIn('rouge', metrics)
        self.assertIn('bleu', metrics)
    
    def test_batch_metrics(self):
        """Test batch metrics calculation"""
        # Create batch of predictions and references
        predictions = [self.good_prediction, self.bad_prediction, self.unrelated_prediction]
        references = [self.reference, self.reference, self.reference]
        
        # Calculate batch metrics
        batch_metrics = self.metrics.calculate_batch_metrics(predictions, references)
        
        # Check structure of batch metrics
        self.assertIn('rouge', batch_metrics)
        self.assertIn('bleu', batch_metrics)
        
        # Check ROUGE metrics structure
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            self.assertIn(rouge_type, batch_metrics['rouge'])
            for score_type in ['precision', 'recall', 'fmeasure']:
                self.assertIn(score_type, batch_metrics['rouge'][rouge_type])
                self.assertIn('mean', batch_metrics['rouge'][rouge_type][score_type])
                self.assertIn('std', batch_metrics['rouge'][rouge_type][score_type])
        
        # Check BLEU metrics structure
        for bleu_type in ['bleu1', 'bleu2', 'bleu4']:
            self.assertIn(bleu_type, batch_metrics['bleu'])
            self.assertIn('mean', batch_metrics['bleu'][bleu_type])
            self.assertIn('std', batch_metrics['bleu'][bleu_type])
    
    def test_mismatched_batch_input(self):
        """Test batch metrics with mismatched input lengths"""
        predictions = [self.good_prediction, self.bad_prediction]
        references = [self.reference]
        
        # Should raise ValueError due to length mismatch
        with self.assertRaises(ValueError):
            self.metrics.calculate_batch_metrics(predictions, references)


if __name__ == '__main__':
    unittest.main()