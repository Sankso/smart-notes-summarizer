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
- **Training Data**: CNN/DailyMail dataset (train[:50000] subset)
- **Training Parameters**: 4 epochs, batch size 2 with gradient accumulation 8, learning rate 3e-4
- **Inference**: Efficient inference with LoRA adapters

## Reasoning Behind Architecture Choices

### 1. Model Selection Rationale

- **Why FLAN-T5-small?**
  - **Balance of Quality and Performance**: With 250M parameters, FLAN-T5-small offers a good balance between summarization quality and computational efficiency
  - **Instruction Tuning**: FLAN (Fine-tuned LAnguage Net) models are already instruction-tuned, making them better suited for task-specific adaptation
  - **Encoder-Decoder Architecture**: T5's encoder-decoder design is particularly effective for text transformation tasks like summarization
  - **Resource Constraints**: Larger models would require significantly more computational resources while providing diminishing returns for our specific use case

- **Why LoRA Fine-tuning?**
  - **Parameter Efficiency**: By training only ~0.5% of the parameters (rank decomposition matrices), we achieve significant memory savings
  - **Training Speed**: Reduced parameter count enables faster training iterations and experimentation
  - **Comparable Performance**: LoRA adaptation achieves similar quality to full fine-tuning for our summarization tasks
  - **Modularity**: Adapters can be swapped or combined without retraining the base model
  - **Future Extensibility**: Additional task-specific adapters can be added without conflicting with summarization capabilities

### 2. Multi-Agent Architecture Justification

- **Why Multiple Agents?**
  - **Separation of Concerns**: Each agent specializes in specific tasks, improving modularity and maintainability
  - **Parallel Development**: Different components can be improved independently
  - **Flexibility**: The system can be easily extended with new capabilities
  - **Targeted Optimization**: Each component can be optimized for its specific task

- **Planner-Executor Pattern Benefits**
  - **Strategic Processing**: The planner can analyze document characteristics before deciding on the optimal processing approach
  - **Resource Optimization**: Different processing strategies can be applied based on document complexity
  - **Quality Control**: The planner can request specific processing parameters based on document analysis

## Logging and Monitoring

- All agent interactions are logged in `docs/logs.md`
- Logs include input details, processing decisions, and outputs
- Processing times and other metadata are tracked

## Testing Infrastructure

- Unit tests for all major components
- Test fixtures and mocks for transformer models
- Evaluation metrics validation

## Utility Components

### 1. Batch Processing System (`scripts/batch_processor.py`)

- Processes multiple documents in sequence
- Maintains consistent settings across batch runs
- Provides aggregated results and statistics
- Supports parallel processing when resources permit

### 2. Helper Utilities

- **Model Management** (`utils/copy_model.py` and `utils/copy_model.ps1`)
  - Facilitates moving model files to the correct locations
  - Cross-platform support with both Python and PowerShell implementations
  - Handles model verification and validation

- **CLI Interface** (`scripts/cli_app.py`)
  - Command-line interface for non-UI usage scenarios
  - Supports automation and scripting
  - Provides the same functionality as the UI in a scriptable form