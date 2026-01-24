
import unittest
from evaluation.metrics import SummarizationMetrics

class TestMetrics(unittest.TestCase):
    def test_rouge(self):
        metrics = SummarizationMetrics()
        ref = "The cat sat on the mat."
        pred = "The cat sat on the mat."
        scores = metrics.calculate_rouge(pred, ref)
        self.assertAlmostEqual(scores['rouge1'].fmeasure, 1.0)
        
        pred_diff = "The dog sat on the mat."
        scores_diff = metrics.calculate_rouge(pred_diff, ref)
        self.assertLess(scores_diff['rouge1'].fmeasure, 1.0)

    def test_bleu(self):
        metrics = SummarizationMetrics()
        ref = "The cat sat on the mat."
        pred = "The cat sat on the mat."
        scores = metrics.calculate_bleu(pred, ref)
        self.assertAlmostEqual(scores['bleu1'], 1.0)
        self.assertAlmostEqual(scores['bleu2'], 1.0)
        self.assertAlmostEqual(scores['bleu4'], 1.0)

if __name__ == '__main__':
    unittest.main()