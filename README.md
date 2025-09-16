# Smart Notes Summarizer

A powerful AI system for summarizing lecture notes, PDFs, and text documents using fine-tuned language models with an intelligent multi-agent architecture.

##🎥 Demo

https://github.com/Sankso/smart-notes-summarizer/issues/1#issue-3422244263

## 📋 Overview

Smart Notes Summarizer is designed to help students and professionals quickly extract key information from lengthy documents. The system employs a sophisticated multi-agent architecture:



- A **Planner** agent that analyzes the input to determine the optimal processing strategy
- An **Executor** agent that generates high-quality summaries using a fine-tuned language model
- A **Keyword Extractor** that identifies key topics and themes from documents
- A **Section Detector** that identifies document structure for more organized summaries

The system uses a fine-tuned FLAN-T5-small model with parameter-efficient techniques (LoRA) to ensure high-quality summaries while maintaining reasonable computational requirements.

## 📊 Model Performance

Our fine-tuned model has been comprehensively evaluated across various text types using our unified evaluation framework that combines both ROUGE and BLEU metrics:

| Text Category | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BLEU-1 | BLEU-2 | BLEU-4 |
|--------------|-----------|-----------|-----------|-------|-------|-------|
| Narrative Text | 0.9200 | 0.8776 | 0.9000 | 0.8952 | 0.8779 | 0.8502 |
| Medium News | 0.7470 | 0.7160 | 0.7470 | 0.7321 | 0.7187 | 0.6893 |
| Technical | 0.6526 | 0.6452 | 0.6526 | 0.6398 | 0.6289 | 0.6105 |
| Abstract | 0.6486 | 0.6389 | 0.6486 | 0.6317 | 0.6215 | 0.6010 |
| Short Factual | 0.5424 | 0.3860 | 0.5085 | 0.5210 | 0.4972 | 0.4715 |
| **Overall Average** | **0.7021** | **0.6527** | **0.6913** | **0.6840** | **0.6688** | **0.6445** |

For complete evaluation results and visualizations, see the [evaluation results directory](evaluation/results/) and the [unified evaluation documentation](docs/unified_evaluation.md).

## 🚀 Features

### Core Features
- **PDF Processing**: Extract and summarize text from PDF files with automatic OCR fallback
- **Multiple Summary Lengths**: Generate short, normal, or long summaries based on your needs
- **Keyword Extraction**: Automatically identify key topics and themes using multiple algorithms
- **Section Detection**: Intelligently identify document sections and summarize each separately
- **Repetition Prevention**: Advanced algorithms to ensure concise, non-repetitive summaries
- **Quality Metrics**: Built-in tools to evaluate summary quality and measure improvements
- **Clean UI**: Intuitive Streamlit interface with summary export capabilities

### Advanced Features
- **Batch Processing**: Process multiple PDFs through scripts and CLI tools
- **User Feedback**: Rating system to collect feedback on summary quality
- **Comprehensive Logging**: Detailed interaction logs for analysis
- **Extensible Architecture**: Modular design for easy customization

## 🧠 Model Fine-tuning

The system uses a FLAN-T5-small model fine-tuned with Low-Rank Adaptation (LoRA) on summarization tasks:

- **Base Model**: google/flan-t5-small
- **Fine-tuning Method**: LoRA (r=16, alpha=32, targeting q and v modules)
- **Dataset**: CNN/DailyMail 3.0.0 (train[:50000] subset)
- **Training Parameters**: 4 epochs, batch size 2 with gradient accumulation 8, learning rate 3e-4
- **Evaluation Metrics**: ROUGE-1/2/L F1 scores and BLEU-1/2/4 scores

To fine-tune your own model, use the provided notebook:

```bash
# Navigate to the finetuning directory
cd finetuning

# Run the training script
python train_lora.py

# OR use the Jupyter notebook
# jupyter notebook train_lora_notebook.ipynb
```

## 💻 Installation & Setup

1. **Clone this repository**:
```bash
git clone https://github.com/yourusername/smart-notes-summarizer.git
cd smart-notes-summarizer
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up your fine-tuned model**:

   a. Place your fine-tuned model files in the `models/lora_weights` directory:
      - adapter_config.json
      - adapter_model.safetensors
      - special_tokens_map.json
      - spiece.model
      - tokenizer_config.json
      - tokenizer.json
   
   b. OR use the provided helper script:
   ```bash
   python utils/copy_model.py path/to/your/model/files
   ```
   
   c. OR use the PowerShell script (Windows):
   ```powershell
   .\utils\copy_model.ps1 path\to\your\model\files
   ```

4. **Run evaluation (optional)**:
```bash
# Run unified evaluation
python evaluation/unified_evaluation.py

# Analyze evaluation results (replace YYYYMMDD_HHMMSS with actual timestamp)
python evaluation/analyze_results.py --result_file evaluation/results/unified_evaluation_YYYYMMDD_HHMMSS.json
```

## 🔧 Usage

### Streamlit UI (Recommended)

The easiest way to use the summarizer is through the intuitive web interface:

```bash
streamlit run ui/app.py
```

The UI provides several options:
- Upload PDF files or paste text directly
- Choose summary length (short, normal, long)
- Enable keyword extraction to identify key topics
- Enable section detection for structured documents
- Export summaries to text files
- Rate summary quality to provide feedback

### Command Line Interface

Process documents directly from the command line:

```bash
# Summarize a PDF
python main.py --pdf "path/to/document.pdf" --output "summary.txt"

# Launch the Streamlit UI
python main.py --ui
```

### Python API for Developers

For programmatic use in your own applications:

```python
from agent.agent import SmartSummarizerAgent

# Initialize the agent
agent = SmartSummarizerAgent()

# Summarize a PDF file with all features
result = agent.summarize_pdf(
    "path/to/document.pdf",
    summary_length="normal",  # Options: "short", "normal", "long"
    extract_keywords=True,
    detect_sections=True
)

# Access summary, keywords and sections
summary = result["summary"]
keywords = result["keywords"]  # List of extracted keywords with scores
sections = result["sections"]  # Dictionary of section summaries

# Process text directly
text_result = agent.summarize_text(
    "Your long text to summarize...",
    summary_length="short",
    extract_keywords=True
)
```

### Batch Processing for Multiple Documents

Process entire directories of PDFs:

```python
from scripts.batch_processor import BatchPDFProcessor

# Initialize the batch processor
processor = BatchPDFProcessor()

# Process a directory of PDFs
results = processor.process_directory(
    "path/to/pdf_folder",
    summary_length="normal",
    extract_keywords=True,
    detect_sections=True
)

# Results contain summary data for each PDF
for pdf_path, result in results.items():
    print(f"Summary for {pdf_path}: {result['summary']}")
```

## 📊 Quality Evaluation

The system includes built-in quality evaluation and repetition detection:

- **Repetition Score**: Measures and reduces repetitive content in summaries
- **Unique Word Ratio**: Ensures vocabulary diversity in generated content
- **ROUGE scores**: Evaluates overlap with reference summaries (when available)
- **BLEU scores**: Evaluates n-gram precision compared to reference summaries

For formal evaluation, run:
```bash
python evaluation/unified_evaluation.py
```

This generates comprehensive evaluation results and stores them in the `evaluation/results` directory with timestamps for easy tracking.

## 🧪 Testing and Validation

Run tests to ensure everything is working correctly:

```bash
# Run all tests
python -m pytest tests/

# Test model loading
python test_model_loading.py
```

## 👨‍💻 Project Structure

```
smart-notes-summarizer/
│
├── agent/                      # Core summarization components
│   ├── agent.py                # Main agent coordinator
│   ├── executor.py             # Summary generation with LLM
│   ├── keyword_extractor.py    # Topic and keyword identification
│   ├── pdf_processor.py        # PDF text extraction
│   ├── planner.py              # Text analysis and strategy planning
│   └── section_detector.py     # Document structure analysis
│
├── ui/                         # User interface
│   ├── app.py                  # Streamlit web application
│   └── test_app.py             # UI tests
│
├── scripts/                    # Helper scripts and integration tools
│   ├── batch_processor.py      # Multiple document processing
│   ├── cli_app.py              # Command-line interface
│   ├── process_pdfs.py         # PDF processing utilities
│   └── ui_integration.py       # Integration with external UIs
│
├── models/                     # Model storage
│   └── lora_weights/           # Fine-tuned model weights
│
├── evaluation/                 # Quality evaluation tools
│   ├── unified_evaluation.py   # Unified evaluation script for ROUGE and BLEU
│   ├── analyze_results.py      # Analysis tools for evaluation results
│   ├── evaluate.py             # Basic evaluation pipeline
│   └── results/                # Evaluation results and visualizations
│
├── finetuning/                 # Model fine-tuning tools
│   ├── download_model.py       # Download base models
│   ├── train_lora.py           # LoRA fine-tuning script
│   ├── train_lora_notebook.ipynb # Jupyter notebook for fine-tuning
│   └── inference.py            # Model inference testing
│
├── tests/                      # Test suite
│   ├── test_agent.py           # Tests for agent functionality
│   └── test_metrics.py         # Tests for evaluation metrics
│
├── utils/                      # Utility functions
│   ├── copy_model.py           # Model setup helper
│   └── copy_model.ps1          # PowerShell model setup helper
│
├── docs/                       # Documentation
│   ├── architecture.md         # System architecture details
│   └── logs.md                 # Interaction logs
│
├── exports/                    # Default location for exported summaries
│
├── main.py                     # Main entry point
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── README_INTEGRATION.md       # Integration documentation
├── RUNNING.md                  # Instructions for running the application
├── USER_GUIDE.md               # User guide and usage instructions
├── test_model_loading.py       # Test script for model loading
└── notebook28d9eb154b (1).ipynb # Additional development notebook
```

## 🏗️ System Architecture

The Smart Notes Summarizer uses a multi-agent architecture to process documents:

1. **Document Processing Flow:**
   - Input (PDF or text) → Text Extraction → Analysis → Summary Generation → Post-processing → Output

2. **Key Components:**
   - **Agent Coordinator** (`agent.py`): Orchestrates the entire process and manages other components
   - **PDF Processor** (`pdf_processor.py`): Extracts text from PDFs with fallback to OCR for images/scans
   - **Planner** (`planner.py`): Analyzes text complexity and determines optimal processing strategy
   - **Executor** (`executor.py`): Generates summaries using the fine-tuned model with anti-repetition techniques
   - **Keyword Extractor** (`keyword_extractor.py`): Uses multiple algorithms to identify key topics
   - **Section Detector** (`section_detector.py`): Identifies document sections for structured summaries

3. **Model Architecture:**
   - Base model: FLAN-T5-small (250M parameters)
   - Fine-tuning: LoRA (Low-Rank Adaptation)
   - Inference: CPU-compatible with GPU acceleration when available
   - Input limit: 10,000 characters per summary (configurable)

4. **Post-processing Pipeline:**
   - Repetition detection and elimination
   - Formatting and structural improvements
   - Quality metrics calculation

## ✨ Future Improvements

Potential enhancements for future versions:

- Multi-language support
- Document comparison and cross-referencing
- Interactive summaries with expandable sections
- Integration with note-taking applications
- Cloud-based API for remote processing

## 📝 Author

- **Name**: Sanskriti Pal
- **University**: IIT Goa
- **Department**: BTech

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
