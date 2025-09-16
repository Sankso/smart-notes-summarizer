# Evaluation Results

This directory contains evaluation results from testing the Smart Notes Summarizer model.

## Directory Contents

- **JSON Results**: Raw evaluation metrics data in JSON format
  - Format: `unified_evaluation_[TIMESTAMP].json` - Complete evaluation metrics with both ROUGE and BLEU scores
  - Format: `bleu_scores_[TIMESTAMP].json` - BLEU-specific evaluation metrics
  
- **Markdown Reports**: Human-readable evaluation reports
  - Format: `unified_report_[TIMESTAMP].md` - Complete evaluation report with ROUGE and BLEU analysis
  - Format: `bleu_report_[TIMESTAMP].md` - BLEU-focused evaluation report
  
- **Plots Directory**: Visual representations of model performance
  - `rouge1_by_category.png` - ROUGE-1 scores across different text categories
  - `rouge1_vs_rouge2.png` - Comparison of ROUGE-1 and ROUGE-2 scores
  - `bleu1_by_category.png` - BLEU-1 scores across different text categories
  - `rouge1_vs_bleu1.png` - Comparison of ROUGE-1 and BLEU-1 scores

## How to Read the Results

### JSON Files

The JSON files contain structured evaluation data including:
- Model information (base model, weights directory, etc.)
- Text samples used for evaluation
- Generated summaries with different parameter sets
- Metrics for each summary (ROUGE scores, BLEU scores, lexical diversity, etc.)
- Aggregate statistics by category and parameter set

### Markdown Reports

The markdown reports provide a human-readable analysis of the evaluation results, including:
- Overall model performance across different text categories
- Comparison of ROUGE and BLEU metrics
- Sample summaries with their associated metrics
- Key observations and recommendations
- Strengths and weaknesses of the model

### Visualizations

The plots visualize different aspects of model performance:
- Performance comparison across different text categories
- Correlation between different evaluation metrics (ROUGE vs BLEU)
- Impact of different parameter settings on summarization quality
- Comparative analysis across text types

## Running New Evaluations

The recommended way to run evaluations is using the updated `run_evaluation.py` script:

```bash
# Complete unified evaluation (ROUGE + BLEU)
python -m evaluation.run_evaluation --mode unified

# Legacy evaluation on CNN/DailyMail dataset
python -m evaluation.run_evaluation --mode legacy

# Clean up old evaluation files without running evaluation
python -m evaluation.run_evaluation --mode clean
```

Additional options:
- `--model`: Base model name (default: "google/flan-t5-small")
- `--lora_dir`: Directory containing LoRA weights (default: "./models/lora_weights")
- `--keep`: Number of latest files to keep for each type (default: 3)
- `--no-clean`: Skip cleaning up old files

New evaluation results will be saved with a timestamp to avoid overwriting previous results, and older files will be automatically cleaned up.

## Latest Results

The most recent unified evaluation shows:

- **ROUGE Metrics**:
  - Average ROUGE-1 F1 score: 0.7021 (default parameters)
  - Best performance on narrative text: 0.9200 ROUGE-1 F1
  - Most challenging text type: abstract text (0.6486 ROUGE-1 F1)

- **BLEU Metrics**:
  - Average BLEU-1 score: 0.6840 (default parameters)
  - Average BLEU-4 score: 0.6445 (default parameters)
  - Strong correlation between ROUGE-1 and BLEU-1 scores across text types

- **Parameter Impact**:
  - Parameter tuning significantly impacts results
  - "more_focused" parameters generally yield higher ROUGE scores
  - "more_diverse" parameters tend to produce better BLEU-4 scores for narrative text

The unified evaluation approach provides a more comprehensive picture of model performance by combining both recall-oriented (ROUGE) and precision-oriented (BLEU) metrics.