# PDF Batch Processing Script

This script allows you to batch process PDF files using the fine-tuned Flan-T5-small model (with LoRA weights) for text summarization.

## Features

- Loads the fine-tuned model and tokenizer
- Extracts text from PDF files
- Automatically splits long PDF text into chunks (max 512 tokens per chunk)
- Summarizes each chunk and combines into a final summary
- Evaluates generated summaries using ROUGE metrics if reference summaries are provided
- Creates detailed interaction logs for each PDF
- Saves logs as JSON files
- Processes multiple PDFs in a folder

## Usage

To process a single PDF file:

```bash
python scripts/process_pdfs.py --input path/to/your/document.pdf
```

To process all PDF files in a directory:

```bash
python scripts/process_pdfs.py --input path/to/pdf/directory
```

To evaluate against a reference summary:

```bash
python scripts/process_pdfs.py --input path/to/your/document.pdf --reference path/to/reference/summary.txt
```

## Output

The script generates:

1. Summaries for each processed PDF file
2. JSON logs with processing details, including:
   - PDF filename
   - Extracted text (snippet)
   - Generated summary
   - Evaluation metrics (if applicable)
   - Processing time

All logs are saved in the `logs` directory.

## Requirements

- Python 3.6+
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- pdfplumber
- rouge-score

These dependencies are already included in the Smart Notes Summarizer environment.