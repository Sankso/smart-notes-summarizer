# Smart Notes Summarizer Agent - Architecture

## Overview

The Smart Notes Summarizer Agent is designed with a modular architecture following a multi-agent approach. The system analyzes lecture notes and PDF documents to generate concise summaries using parameter-efficient fine-tuned language models.

## System Architecture

```
                    +------------------------+
                    |   Smart Notes Agent    |
                    +------------------------+
                            |
                            v
         +------------------+------------------+
         |                  |                  |
         v                  v                  v
+----------------+  +----------------+  +----------------+
| Planner Agent  |  |  Executor Agent |  | PDF Processor  |
+----------------+  +----------------+  +----------------+
                            |                  
                            v                  
                  +------------------+         
                  |   LoRA Fine-tuned|         
                  |    FLAN-T5 Model |         
                  +------------------+         
                            |                  
                            v                  
                  +------------------+         
                  |  Streamlit UI    |         
                  +------------------+         
```

## Core Components

### 1. Multi-Agent System

The system implements a multi-agent architecture:

1. **Planner Agent** (`agent/planner.py`)
   - Analyzes input text to determine the optimal processing approach
   - Uses text length, complexity, and structure to decide between summarization or rewriting
   - Features text statistics extraction and complexity scoring

2. **Executor Agent** (`agent/executor.py`)
   - Handles the actual text transformation based on the planner's decision
   - Interfaces with the fine-tuned language model to generate summaries or rewrites
   - Supports Parameter-Efficient Fine-Tuning (LoRA adapters)
   - Provides multiple parameter sets for different summarization styles

3. **Main Agent** (`agent/agent.py`)
   - Coordinates the end-to-end process
   - Handles PDF processing via the PDF Processor
   - Manages interaction logging

4. **Keyword Extractor** (`agent/keyword_extractor.py`)
   - Extracts key terms and concepts from the text
   - Uses techniques like TF-IDF, RAKE, and YAKE for keyword identification
   - Provides context for summaries

5. **Section Detector** (`agent/section_detector.py`)
   - Identifies logical sections within documents
   - Enables section-specific processing and summarization
   - Preserves document structure in summaries

### 2. Model Training & Evaluation

1. **Fine-tuning Pipeline** (`finetuning/`)
   - LoRA (Low-Rank Adaptation) fine-tuning on google/flan-t5-small
   - Training on CNN/DailyMail summarization dataset
   - Parameter-efficient approach to reduce computational requirements
   - Implemented in `train_lora.py` and `train_lora_notebook.ipynb`
   - Model downloading functionality in `download_model.py`
   - Inference capabilities through `inference.py`

2. **Evaluation Framework** (`evaluation/`)
   - Comprehensive evaluation across different text categories
   - Multiple parameter settings to test model versatility
   - ROUGE and BLEU metrics calculation for summary quality assessment
   - Visualization generation for performance analysis
   - Components:
     - `unified_evaluation.py`: Comprehensive evaluation across text types
     - `evaluate.py`: Core evaluation functionality
     - `analyze_results.py`: Analysis and cleanup utilities for evaluation results
   - Provides detailed reports and visualizations

3. **PDF Processor** (`agent/pdf_processor.py`)
   - Extracts text from PDF documents
   - Handles scanned documents via OCR when needed
   - Provides advanced text extraction capabilities
   - Preserves document structure where possible

### 3. Fine-tuning Pipeline

The system includes a complete fine-tuning pipeline for customizing the model:

1. **LoRA Training** (`finetuning/train_lora.py` and `finetuning/train_lora_notebook.ipynb`)
   - Implements Parameter-Efficient Fine-Tuning
   - Trains only a small set of adapter parameters (~0.5% of full model)
   - Supports efficient training on consumer hardware
   - Jupyter notebook version for interactive experimentation

2. **Model Download** (`finetuning/download_model.py`)
   - Facilitates downloading pre-trained models
   - Handles Hugging Face model repository integration
   - Manages model caching and version control

3. **Inference Module** (`finetuning/inference.py`)
   - Handles model loading and inference
   - Supports both base and LoRA-adapted models
   - Provides batched processing for efficiency

### 4. Evaluation System

The evaluation system provides quantitative metrics:

1. **Unified Evaluation** (`evaluation/unified_evaluation.py`)
   - Comprehensive evaluation across different text categories
   - Implements both ROUGE and BLEU metrics for summary quality
   - Generates visualizations for result analysis
   - Produces detailed markdown reports

2. **Evaluator Class** (`evaluation/evaluate.py`)
   - Implements the `SummarizerEvaluator` class
   - Runs inference on test datasets
   - Calculates and reports quality metrics
   - Formats results in human-readable format

3. **Results Analysis** (`evaluation/analyze_results.py`)
   - Identifies redundant evaluation files
   - Provides summary statistics of evaluation runs
   - Helps maintain clean evaluation history

### 5. User Interface

The UI provides a user-friendly way to interact with the system:

1. **Streamlit App** (`ui/app.py`)
   - PDF upload and processing functionality
   - Direct text input interface
   - Configuration options for summarization parameters
   - Summary display with keyword highlighting
   - Export functionality for saving results
   - User feedback collection for model improvement
   - Batch processing capabilities for multiple documents

2. **UI Testing** (`ui/test_app.py`)
   - Test suite for UI functionality
   - Ensures reliable user experience
   - Validates integration with backend components

## Data Flow

1. **Input Processing**:
   - PDF documents are processed by the PDF Processor to extract text
   - Text input is used directly

2. **Analysis**:
   - The Planner Agent analyzes the input to determine the optimal approach
   - Text statistics and complexity scores are calculated

3. **Generation**:
   - The Executor Agent uses the fine-tuned model to generate the output
   - Either a concise summary (for long texts) or a rewritten version (for short texts)

4. **Output & Feedback**:
   - Results are displayed to the user
   - Optional export to text files
   - User ratings can be collected

## Model Details

- **Base Model**: FLAN-T5-small (google/flan-t5-small)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: Summarization datasets (processed via dataset_preparation.py)
- **Inference**: Efficient inference with LoRA adapters

## Logging and Monitoring

- All agent interactions are logged in `docs/logs.md`
- Logs include input details, processing decisions, and outputs
- Processing times and other metadata are tracked

## Testing Infrastructure

- Unit tests for all major components
- Test fixtures and mocks for transformer models
- Evaluation metrics validation