# Unified Evaluation Framework

This document describes the unified evaluation framework implemented for the Smart Notes Summarizer project, which combines both ROUGE and BLEU metrics for comprehensive model assessment.

## Overview

The unified evaluation framework provides a comprehensive approach to evaluating the summarization model's performance. Rather than relying on a single metric type, it combines both ROUGE and BLEU scores to provide a more complete picture of summary quality.

### Key Components:

1. **Metrics Calculation**
   - ROUGE metrics (precision, recall, and F1 scores)
   - BLEU metrics (n-gram overlap scores)
   - Summary statistics (length, sentence count, lexical diversity)

2. **Multi-parameter Evaluation**
   - Default parameters (balanced approach)
   - More diverse parameters (increased novelty)
   - More focused parameters (improved conciseness)

3. **Text Type Diversity**
   - Short factual texts
   - Medium-length news articles
   - Technical documentation
   - Narrative/creative content
   - Academic papers

4. **Visualization**
   - ROUGE scores by category
   - BLEU scores by category
   - ROUGE vs BLEU comparison plots

## Metrics Explained

### ROUGE Metrics

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) focuses on measuring the overlap between the generated summary and reference summaries:

- **ROUGE-1**: Unigram (single word) overlap
- **ROUGE-2**: Bigram (two consecutive words) overlap
- **ROUGE-L**: Longest Common Subsequence overlap

ROUGE scores provide three values:
- **Precision**: How much of the generated summary appears in the reference
- **Recall**: How much of the reference appears in the generated summary
- **F1**: Harmonic mean of precision and recall

### BLEU Metrics

BLEU (Bilingual Evaluation Understudy) was originally designed for machine translation but works well for summarization too:

- **BLEU-1**: Unigram precision
- **BLEU-2**: Bigram precision
- **BLEU-4**: 4-gram precision

BLEU focuses more on precision than recall, complementing ROUGE's recall-oriented approach.

## Usage

### Running the Unified Evaluation

To run the unified evaluation, use the `unified_evaluation.py` script directly:

```bash
python -m evaluation.unified_evaluation --model "google/flan-t5-small" --lora_dir "./models/lora_weights" --output_dir "./evaluation/results"
```

### Additional Options

- `--model`: Base model name (default: "google/flan-t5-small")
- `--lora_dir`: Directory containing LoRA weights (default: "./models/lora_weights")
- `--output_dir`: Directory to save evaluation results (default: "./evaluation/results")

### Output Files

The evaluation produces several output files:

1. **JSON Results**: `unified_evaluation_[timestamp].json`
   - Contains all raw metrics and generated summaries

2. **Markdown Report**: `unified_report_[timestamp].md`
   - Formatted report with analysis and observations

3. **Visualizations**: in the `results/plots/` directory
   - ROUGE and BLEU scores by category
   - Comparison plots between metrics

## Implementation Details

The unified evaluation uses a custom `MetricsCalculator` class that implements both ROUGE and BLEU calculations. For ROUGE, it leverages the `rouge_score` library, while for BLEU, it implements a custom n-gram overlap calculation.

The framework generates summaries using three parameter sets to evaluate how different generation strategies affect summary quality. It also tests the model across various text types to assess its versatility.

## Managing Evaluation Results

For managing evaluation results, you can use the `analyze_results.py` script to identify redundant files and clean up the results directory:

```bash
python -m evaluation.analyze_results --dir "./evaluation/results" --all
```

To delete redundant files:

```bash
python -m evaluation.analyze_results --dir "./evaluation/results" --delete
```

This script helps maintain a clean results directory by identifying and removing outdated evaluation files.

## Future Improvements

Potential future enhancements to the evaluation framework include:

1. Adding more semantic evaluation metrics like BERTScore
2. Implementing human evaluation integration
3. Expanding the text type coverage for more specialized domains
4. Comparative analysis with other summarization models