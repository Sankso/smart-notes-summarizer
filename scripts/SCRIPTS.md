# Smart Notes Summarizer CLI and Integration Modules

This directory contains scripts for extending the Smart Notes Summarizer beyond the UI, including batch processing capabilities and integration with other tools.

## Available Scripts

### 1. `process_pdfs.py`

A script for batch processing PDF files using the fine-tuned model.

```bash
python scripts/process_pdfs.py --input path/to/your/document.pdf
```

See [PDF Batch Processing documentation](README.md) for details.

### 2. `batch_processor.py`

A class that encapsulates the PDF batch processing functionality, designed to be imported and used in other Python code.

```python
from scripts.batch_processor import BatchPDFProcessor

processor = BatchPDFProcessor()
result = processor.process_pdf("document.pdf")
print(result["summary"])
```

### 3. `cli_app.py`

A command-line interface for the Smart Notes Summarizer, providing easy access to all features.

```bash
python -m scripts.cli_app --pdf_path path/to/document.pdf --summary_length short --extract_keywords
```

#### Arguments

- `--pdf_path`: Path to a PDF file or directory containing PDFs
- `--mode`: Processing mode (default: single)
  - `single`: Process a single PDF file
  - `batch`: Process all PDFs in a directory
- `--output_dir`: Directory to save results (default: exports)
- `--summary_length`: Length of summary (default: normal)
  - `short`: Generate a shorter summary
  - `normal`: Generate a standard-length summary
  - `long`: Generate a longer, more detailed summary
- `--extract_keywords`: Flag to extract keywords from the document

### 4. `ui_integration.py`

Integration module to connect the batch processor with the Smart Notes Summarizer Streamlit UI.

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

## Requirements

All scripts use the same dependencies as the main Smart Notes Summarizer application, which are listed in the project's `requirements.txt`.