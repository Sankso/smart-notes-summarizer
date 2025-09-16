# Troubleshooting Guide

This document provides solutions for common issues you might encounter when using the Smart Notes Summarizer.

## Installation Issues

### Missing Dependencies

**Issue**: Error when installing requirements or running the application due to missing dependencies.

**Solution**:
1. Ensure you have Python 3.9+ installed:
   ```bash
   python --version
   ```

2. Install required system packages:
   - Windows: Install Visual C++ Build Tools
   - Linux: `sudo apt-get install build-essential python3-dev`
   - macOS: `brew install libomp`

3. Try installing requirements with:
   ```bash
   pip install -r requirements.txt --no-cache-dir
   ```

### CUDA/GPU Issues

**Issue**: Warnings about CUDA not being available or errors related to GPU.

**Solution**:
1. The application works fine with CPU, but for better performance:
   - Install compatible CUDA toolkit for your GPU
   - Install appropriate PyTorch version matching your CUDA version

2. Verify CUDA is working:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

## Model Loading Problems

### Model Not Found

**Issue**: Error indicating model weights cannot be found.

**Solution**:
1. Ensure the model weights are in the correct location:
   ```
   smart-notes-summarizer/
   └── models/
       └── lora_weights/
           ├── adapter_config.json
           ├── adapter_model.safetensors
           └── ...
   ```

2. Run the model download script if needed:
   ```bash
   python finetuning/download_model.py
   ```

### Out of Memory Errors

**Issue**: Process crashes with CUDA out of memory or system memory errors.

**Solution**:
1. Reduce batch size in configuration:
   - Edit `smart-notes-summarizer/agent/executor.py`
   - Find and lower the batch_size parameter

2. Process smaller chunks of text at a time:
   - Split large documents into smaller parts
   - Process each part separately

## PDF Processing Issues

### Text Extraction Failures

**Issue**: No text extracted from PDF or poor quality extraction.

**Solution**:
1. Check if PDF contains actual text (not just images):
   - Try copying text directly from the PDF
   - If you can't select text, the PDF contains only images

2. For image-only PDFs:
   - Install OCR components: `pip install pytesseract`
   - Install Tesseract OCR engine for your OS
   - Enable OCR in configuration

### Specific PDF Format Issues

**Issue**: Errors with specific PDF files or unusual formats.

**Solution**:
1. Try alternative extraction mode:
   ```bash
   python scripts/process_pdfs.py --input_file problematic.pdf --extraction_mode fallback
   ```

2. Convert the PDF to a different format:
   - Use online tools to convert to a standard PDF format
   - Save as PDF/A format which is more standardized

## Summary Quality Issues

### Repetitive Content

**Issue**: Summaries contain repetitive phrases or sentences.

**Solution**:
1. Enable the enhanced repetition detection:
   ```python
   result = agent.summarize_pdf(pdf_path, repetition_reduction='high')
   ```

2. Try different summary lengths:
   - Short summaries often have less repetition
   - Set `summary_length='short'` in the API call

### Irrelevant Content

**Issue**: Summaries focus on wrong aspects or include irrelevant details.

**Solution**:
1. Try enabling section detection:
   ```python
   result = agent.summarize_pdf(pdf_path, detect_sections=True)
   ```

2. For technical documents, use the research-focused model:
   ```python
   agent = SmartSummarizerAgent(model_profile="research")
   ```

## UI Application Issues

### Streamlit Not Starting

**Issue**: Error when starting the Streamlit application.

**Solution**:
1. Check for port conflicts:
   - Another application might be using port 8501
   - Kill the conflicting process or specify a different port:
     ```bash
     streamlit run ui/app.py --server.port 8502
     ```

2. Verify Streamlit installation:
   ```bash
   pip install --upgrade streamlit
   ```

### Slow Performance in UI

**Issue**: UI becomes unresponsive during summarization.

**Solution**:
1. Set processing timeout:
   - Edit `ui/app.py`
   - Find and adjust the timeout parameter

2. Process files offline first:
   ```bash
   python scripts/process_pdfs.py --input_file document.pdf --output_file summary.txt
   ```

## Batch Processing Issues

### Processing Hangs

**Issue**: Batch processing appears to hang or runs extremely slowly.

**Solution**:
1. Reduce parallel processing:
   ```bash
   python scripts/batch_processor.py --input_dir ./docs --output_dir ./out --parallel 1
   ```

2. Enable detailed logging to identify problematic files:
   ```bash
   python scripts/batch_processor.py --input_dir ./docs --output_dir ./out --log_file batch.log --log_level debug
   ```

## Getting Help

If you continue to experience issues:

1. Check the logs in the `logs/` directory for detailed error information
2. Search for similar issues in the project documentation
3. Try running the test scripts to verify basic functionality:
   ```bash
   python test_model_loading.py
   ```

## Reporting Issues

When reporting issues, please include:
- Exact error messages or screenshots
- Steps to reproduce the problem
- System information (OS, Python version, etc.)
- Log files if available