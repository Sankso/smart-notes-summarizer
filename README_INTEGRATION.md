# Smart Notes Summarizer Integration

This package provides a comprehensive Python solution for integrating the Smart Notes Summarizer system with batch processing capabilities and command-line interfaces. These additions extend the existing Streamlit-based UI, allowing for more flexible usage scenarios.

## New Components

### 1. Batch Processing

Process multiple PDFs automatically using the fine-tuned model:

```python
from scripts.batch_processor import BatchPDFProcessor

processor = BatchPDFProcessor()
results = processor.process_directory("path/to/pdf_folder")
```

### 2. Command-Line Interface

Run the summarizer directly from the command line:

```bash
python -m scripts.cli_app --pdf_path path/to/document.pdf --summary_length short --extract_keywords
```

### 3. UI Integration

Integrate the batch processing capabilities with the existing UI:

```python
from scripts.ui_integration import SmartNotesSummarizerIntegration

integration = SmartNotesSummarizerIntegration()
result = integration.process_pdf(
    "document.pdf",
    summary_length="normal",
    extract_keywords=True,
    detect_sections=True
)
```

## Features

- Process single PDFs or entire directories
- Extract keywords from documents
- Generate summaries of different lengths (short, normal, long)
- Detect and summarize document sections
- Save processing logs and results for later analysis
- Evaluate summary quality with comprehensive metrics (ROUGE and BLEU)

## Usage

See the provided example script for detailed usage examples:

```bash
python -m scripts.examples
```

## Requirements

All scripts use the same dependencies as the main Smart Notes Summarizer application.