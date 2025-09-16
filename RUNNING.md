# Running the Smart Notes Summarizer with Your Fine-tuned Model

This guide will help you run the Smart Notes Summarizer with your existing fine-tuned model.

## Step 1: Place Your Fine-tuned Model Files

Place your fine-tuned model files in the `models/lora_weights` directory. You can use one of these methods:

### Method A: Manual Copy
Simply copy your model files directly into the `models/lora_weights` folder.

### Method B: Use the Python Helper Script
```
python utils/copy_model.py path/to/your/model/files
```

### Method C: Use the PowerShell Script (Windows)
```
.\utils\copy_model.ps1 path\to\your\model\files
```

## Step 2: Run the Application

Once your model files are in place, you can run the application using Streamlit:

```
streamlit run ui/app.py
```

This will start the web interface, which you can access in your browser at http://localhost:8501

## Step 3: Using the Interface

1. **Upload a PDF**: Use the file uploader to select a PDF document
2. **Or Enter Text**: Type or paste text directly into the text area
3. **Generate Summary**: Click the "Generate Summary" button
4. **Export Summary**: Use the export button to save the summary as a text file
5. **Rate Summary**: Provide feedback on summary quality using the rating system

## Troubleshooting

If you encounter issues with your model:

- Check console output for error messages
- Verify that all required model files are present in the `models/lora_weights` directory
- Ensure your model was fine-tuned from a FLAN-T5 base model

## Need Help?

If you need additional assistance, please refer to the project documentation in the `docs` directory.