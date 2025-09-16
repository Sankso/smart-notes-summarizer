# User Guide: Smart Notes Summarizer

This guide provides step-by-step instructions for using the Smart Notes Summarizer application.

## 1. Getting Started

### Starting the Application

The easiest way to use Smart Notes Summarizer is through its Streamlit interface:

```bash
cd smart-notes-summarizer
streamlit run ui/app.py
```

This will open a web browser with the application interface.

## 2. PDF Summarization

### Uploading a PDF

1. In the application, select the **PDF Summarization** tab
2. Click the **Browse files** button to upload your PDF
3. Wait for the file to upload (you'll see file information displayed)

### Summary Options

After uploading, configure your summary preferences:

1. **Summary Length**:
   - **Short**: 1-2 sentences, very concise (50-75 words)
   - **Normal**: Standard summary (150-200 words)
   - **Long**: Detailed summary with more information (250-300 words)

2. **Additional Features**:
   - **Extract Keywords**: Identifies key topics in the document
   - **Detect & Summarize Sections**: Analyzes the document structure and provides section-by-section summaries

3. Click **Generate Summary** to process the document

### Viewing Results

The application will display:
- The main summary at the top
- Key topics (if extracted) as colored tags
- Section summaries (if enabled) in expandable sections

### Exporting

To save your summary:
1. Click **Export Summary to Text File**
2. The file will be saved to the `exports` folder in the project directory

## 3. Text Summarization

To summarize text without a PDF:

1. Select the **Text Summarization** tab
2. Paste or type your text in the text area
3. Configure summary options as described above
4. Click **Generate Summary**

## 4. Understanding Summary Quality

The application shows quality metrics for each summary:

- **Quality Score**: Overall measure of summary quality
- **Unique Words**: Percentage of unique words (higher is better)
- **Repetition**: Measure of repetitive content (lower is better)

## 5. Tips for Best Results

- **PDF Quality**: Cleaner PDFs with searchable text work best
- **Text Length**: The system works best with input between 500-5000 words
- **Section Detection**: Works best with documents that have clear headings
- **Summary Length**: Choose based on your needs:
  - Short: Quick overview of main points
  - Normal: Balanced summary for general understanding
  - Long: Comprehensive coverage of content

## 6. Batch Processing

For processing multiple PDFs at once, use the batch processing script:

```bash
python -m scripts.batch_processor --input_dir "path/to/pdfs" --output_dir "path/to/output"
```

## 7. Troubleshooting

Common issues and solutions:

- **PDF text extraction fails**: The document may be scanned or have security restrictions
- **Summary too short**: Try using the "long" summary option
- **Out of memory errors**: Reduce the size of PDFs or use the CLI tool instead

## 8. Getting Help

If you encounter any issues or have questions, please:
1. Check the documentation in the `docs` folder
2. Run the test scripts to verify your installation
3. Contact the project author with details about your problem