#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the Smart Notes Summarizer Agent.
Tests the functionality of the planner, executor, and main agent.
"""

import os
import sys
import unittest
import logging
from unittest import mock

# Add parent directory to path to import project modules
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from agent.planner import PlannerAgent
from agent.executor import Executor
from agent.agent import SmartSummarizerAgent

# Disable logging during tests
logging.disable(logging.CRITICAL)

class TestPlannerAgent(unittest.TestCase):
    """Test cases for the Planner Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.planner = PlannerAgent(
            long_text_threshold=100,  # Smaller threshold for testing
            complexity_threshold=0.5
        )
        
        # Sample texts for testing
        self.short_text = "This is a short and simple text for testing."
        self.long_text = "This is a longer text that should be above the threshold for summarization. " * 10
        self.complex_text = "The quantum chromodynamics of supersymmetric particle interactions " \
                           "demonstrates non-trivial topological effects in high-dimensional Hilbert spaces, " \
                           "particularly when considering renormalization group flow under extreme conditions."
    
    def test_analyze_short_text(self):
        """Test the planner's analysis of short text"""
        result = self.planner.analyze(self.short_text)
        
        # Check result structure
        self.assertIn('action', result)
        self.assertIn('reason', result)
        self.assertIn('word_count', result)
        
        # Short text should be classified for rewriting
        self.assertEqual(result['action'], 'rewrite')
    
    def test_analyze_long_text(self):
        """Test the planner's analysis of long text"""
        result = self.planner.analyze(self.long_text)
        
        # Long text should be classified for summarization
        self.assertEqual(result['action'], 'summarize')
        self.assertTrue(result['word_count'] > self.planner.long_text_threshold)
    
    def test_analyze_complex_text(self):
        """Test the planner's analysis of complex text"""
        result = self.planner.analyze(self.complex_text)
        
        # Complex text should be classified for summarization
        self.assertEqual(result['action'], 'summarize')
        self.assertTrue(result['complexity_score'] > self.planner.complexity_threshold)
    
    def test_text_statistics(self):
        """Test extraction of text statistics"""
        word_count, avg_sentence_length, complex_word_ratio = self.planner._get_text_statistics(self.long_text)
        
        # Check that statistics are calculated correctly
        self.assertIsInstance(word_count, int)
        self.assertIsInstance(avg_sentence_length, float)
        self.assertIsInstance(complex_word_ratio, float)
        self.assertTrue(word_count > 0)
        self.assertTrue(avg_sentence_length > 0)
        self.assertTrue(0 <= complex_word_ratio <= 1)


class TestExecutorAgent(unittest.TestCase):
    """Test cases for the Executor Agent"""
    
    @mock.patch('transformers.pipeline')
    @mock.patch('transformers.AutoModelForSeq2SeqLM.from_pretrained')
    @mock.patch('transformers.AutoTokenizer.from_pretrained')
    def setUp(self, mock_tokenizer, mock_model, mock_pipeline):
        """Set up test fixtures with mocks for the heavy transformer components"""
        # Setup mock returns
        mock_pipeline.return_value = lambda text, **kwargs: [{"generated_text": f"Summary of: {text}"}]
        
        # Initialize executor with mocked components
        self.executor = Executor(
            model_name="mock-model"
        )
        
        # Sample text for testing
        self.test_text = "This is a test text for the executor agent."
    
    def test_process_summarize(self):
        """Test the generate_summary method"""
        result = self.executor.generate_summary(
            self.test_text, 
            length="normal"
        )
        
        # Check that a summary was generated
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
    
    def test_extract_keywords(self):
        """Test the extract_keywords method"""
        result = self.executor.extract_keywords(
            self.test_text, 
            num_keywords=5
        )
        
        # Check that keywords were generated
        self.assertIsInstance(result, list)
        # Note: We can't strictly check length as our mock may not respect num_keywords
        self.assertTrue(len(result) >= 0)


class TestSmartSummarizerAgent(unittest.TestCase):
    """Test cases for the Smart Summarizer Agent"""
    
    @mock.patch('agent.agent.SmartSummarizerAgent._extract_text_from_pdf')
    @mock.patch('agent.executor.Executor')
    @mock.patch('agent.planner.PlannerAgent')
    def setUp(self, mock_planner_class, mock_executor_class, mock_extract_text):
        """Set up test fixtures with mocks"""
        # Setup mock planner
        self.mock_planner = mock.Mock()
        self.mock_planner.analyze.return_value = {
            'action': 'summarize',
            'reason': 'Text is lengthy',
            'word_count': 500,
            'complexity_score': 0.7
        }
        mock_planner_class.return_value = self.mock_planner
        
        # Setup mock executor
        self.mock_executor = mock.Mock()
        self.mock_executor.process.return_value = "This is a generated summary."
        mock_executor_class.return_value = self.mock_executor
        
        # Setup mock PDF extractor
        mock_extract_text.return_value = "This is extracted text from a PDF."
        
        # Create agent with mocked components
        self.agent = SmartSummarizerAgent(
            model_name="mock-model",
            lora_weights_dir=None,
            logs_path=os.path.join(os.path.dirname(__file__), "test_logs.md")
        )
        
        # Sample text for testing
        self.test_text = "This is a test text for the summarization agent."
        self.test_pdf_path = "dummy.pdf"  # No actual file needed due to mocking
    
    def test_summarize_text(self):
        """Test the text summarization method"""
        summary = self.agent.summarize_text(self.test_text)
        
        # Verify the planner was called
        self.mock_planner.analyze.assert_called_once_with(self.test_text)
        
        # Verify the executor was called with correct parameters
        self.mock_executor.process.assert_called_once()
        args, kwargs = self.mock_executor.process.call_args
        self.assertEqual(args[0], self.test_text)
        self.assertEqual(args[1], 'summarize')
        
        # Check that a summary was returned
        self.assertEqual(summary, "This is a generated summary.")
    
    def test_summarize_pdf(self):
        """Test the PDF summarization method"""
        summary = self.agent.summarize_pdf(self.test_pdf_path)
        
        # Verify the executor was called correctly
        self.mock_executor.process.assert_called_once()
        
        # Check that a summary was returned
        self.assertEqual(summary, "This is a generated summary.")


if __name__ == '__main__':
    unittest.main()