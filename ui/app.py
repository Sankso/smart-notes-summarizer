#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit Web Application for Smart Notes Summarizer.

This module implements a user-friendly web interface for the Smart Notes Summarizer
using Streamlit. The application provides features for:

1. Uploading and processing PDF documents (lecture notes, papers, reports)
2. Direct text input for summarization (copy-pasted notes, text content)
3. Configuration options for summary length and additional features
4. Display of results with keywords, section summaries, and export options
5. Batch processing capabilities for multiple documents

The UI is designed to be intuitive and accessible for both technical and
non-technical users, with appropriate feedback during processing and
clear presentation of results.

Usage:
    Run with: `streamlit run ui/app.py`
    Access in browser at: http://localhost:8501
"""

import os
import sys
import time
import logging
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Optional, Any

import streamlit as st
import pandas as pd
from PIL import Image

# Add parent directory to path to import project modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from agent.agent import SmartSummarizerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
SUPPORTED_FORMATS = ["pdf"]
TEMP_DIR = tempfile.gettempdir()
EXPORTS_DIR = os.path.join(parent_dir, "exports")

# Ensure exports directory exists
os.makedirs(EXPORTS_DIR, exist_ok=True)

def initialize_agent() -> SmartSummarizerAgent:
    """
    Initialize and configure the Smart Summarizer Agent with fine-tuned model weights.
    
    This function handles the creation and caching of the agent instance in the
    Streamlit session state to prevent reloading the model on each interaction.
    The function implements a singleton pattern to ensure only one instance
    of the agent is active during the application session.
    
    The agent is configured to use the fine-tuned LoRA weights from the local
    models directory, which provides domain-specific summarization capabilities
    optimized for academic and technical content.
    
    Returns:
        SmartSummarizerAgent: Configured agent instance ready for summarization tasks
        
    Note:
        The first initialization may take several seconds as the model weights are loaded
        A spinner is displayed during initialization to provide visual feedback
    """
    if "agent" not in st.session_state:
        # Initialize agent with custom fine-tuned model
        with st.spinner("Initializing summarizer agent with fine-tuned model..."):
            # Path to the fine-tuned model
            lora_weights_dir = os.path.join(parent_dir, "models", "lora_weights")
            
            # Initialize agent with the fine-tuned model
            st.session_state["agent"] = SmartSummarizerAgent(
                lora_weights_dir=lora_weights_dir
            )
    
    return st.session_state["agent"]

def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to temporary directory.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Path to the saved file
    """
    # Create temp file path with original filename
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    
    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def export_summary(summary: str, filename: str) -> str:
    """
    Export summary to a text file.
    
    Args:
        summary: Summary text to export
        filename: Base filename (without extension)
        
    Returns:
        Path to the exported file
    """
    # Create a valid filename
    safe_filename = "".join([c if c.isalnum() or c in [' ', '.', '-', '_'] else '_' for c in filename])
    safe_filename = safe_filename.replace(' ', '_')
    
    # Add timestamp to ensure uniqueness
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    export_filename = f"{safe_filename}_{timestamp}.txt"
    
    # Full path to export file
    export_path = os.path.join(EXPORTS_DIR, export_filename)
    
    # Write summary to file
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(summary)
    
    return export_path

def display_header():
    """
    Render the application header, title and introductory description.
    
    This function handles the display of the main application title and
    information section at the top of the UI. It introduces the application's
    purpose and capabilities to new users and provides basic context about
    how the system works.
    
    The header includes:
    - Application title with appropriate styling
    - Brief description of the app's purpose and functionality
    - Information about the underlying model being used
    - Basic guidance on how to get started with the application
    """
    st.title("Smart Notes Summarizer")
    
    st.markdown("""
    This app helps you generate concise, high-quality summaries from lecture notes and PDF documents.
    Upload a PDF file or paste your text below to get started.
    
    *Using fine-tuned model from models/lora_weights*
    """)

def display_sidebar():
    """Display and handle sidebar UI elements"""
    st.sidebar.header("About")
    st.sidebar.markdown("""
    **Smart Notes Summarizer** uses AI to create high-quality summaries of academic content.
    
    **Features:**
    - PDF document processing
    - Smart text analysis
    - Fine-tuned AI summarization
    
    **Developer:**
    - Name: Sanskriti Pal
    - University: IIT Goa
    - Department: BTech
    """)
    
    st.sidebar.header("Options")
    show_analysis = st.sidebar.checkbox("Show analysis details", value=False)
    
    # Return sidebar options
    return {
        "show_analysis": show_analysis
    }

def pdf_summarization_tab():
    """
    Handle the PDF upload interface and document summarization workflow.
    
    This function manages the complete PDF processing UI flow:
    1. Provides a file uploader with validation for PDF file type and size limits
    2. Configures summarization options (length, features, section detection)
    3. Processes the uploaded PDF when the user clicks the summarize button
    4. Displays progress feedback during processing
    5. Renders the summarization results with proper formatting
    6. Provides options to export or save the generated summary
    
    The function implements proper error handling for invalid files, extraction
    failures, and processing exceptions to ensure a robust user experience.
    """
    st.header("PDF Summarization")
    
    # File upload widget
    uploaded_file = st.file_uploader("Upload a PDF document", type=SUPPORTED_FORMATS)
    
    if uploaded_file:
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"File size exceeds the {MAX_FILE_SIZE/1024/1024:.1f}MB limit.")
            return
        
        # Display file info
        st.info(f"File: {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB)")
        
        # Options for summary generation
        st.subheader("Summary Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            summary_length = st.radio(
                "Summary Length",
                ["short", "normal", "long"],
                index=1,
                help="Short: 1-2 sentences, Normal: default length, Long: detailed summary"
            )
        
        with col2:
            extract_keywords = st.checkbox("Extract Keywords", value=True, 
                                          help="Extract key topics from the document")
            detect_sections = st.checkbox("Detect & Summarize Sections", value=False,
                                         help="Identify document sections and summarize each separately")
        
        # Process button
        if st.button("Generate Summary", key="pdf_generate_btn"):
            # Initialize agent
            agent = initialize_agent()
            
            # Save uploaded file
            with st.spinner("Saving uploaded file..."):
                file_path = save_uploaded_file(uploaded_file)
            
            # Process PDF
            with st.spinner("Extracting text from PDF..."):
                try:
                    # Process PDF and get summary with options
                    result = agent.summarize_pdf(
                        file_path, 
                        summary_length=summary_length,
                        extract_keywords=extract_keywords,
                        detect_sections=detect_sections
                    )
                    
                    # Display summary
                    st.subheader("Summary")
                    st.write(result["summary"])
                    
                    # Display repetition metrics if available
                    if "repetition_metrics" in result:
                        with st.expander("Summary Quality Metrics"):
                            metrics = result["repetition_metrics"]
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Lower is better for repetition score (0-1)
                                quality_score = 100 - (metrics.get("repetition_score", 0) * 100)
                                st.metric("Quality Score", f"{quality_score:.1f}%")
                                
                            with col2:
                                # Higher is better for unique word ratio (0-1)
                                unique_ratio = metrics.get("unique_word_ratio", 0) * 100
                                st.metric("Unique Words", f"{unique_ratio:.1f}%")
                                
                            with col3:
                                # Lower is better for repeated phrase ratio (0-1)
                                repetition = metrics.get("repeated_phrase_ratio", 0) * 100
                                st.metric("Repetition", f"{repetition:.1f}%", 
                                         delta=f"-{repetition:.1f}%", delta_color="inverse")
                    
                    # Display keywords if available
                    if extract_keywords and "keywords" in result and result["keywords"]:
                        st.subheader("Key Topics")
                        
                        # Create clickable keyword tags with colors
                        keyword_html = ""
                        
                        for kw in result["keywords"][:10]:
                            keyword = kw["keyword"]
                            score = kw["score"]
                            # Higher score = more intense color
                            intensity = min(100, int(score * 100)) if score <= 1 else 100
                            keyword_html += f"""<span style="display: inline-block; 
                                                margin: 2px; padding: 4px 8px; 
                                                background-color: rgba(59, 130, 246, {intensity/100}); 
                                                border-radius: 16px; color: white; 
                                                font-size: 0.9em;">{keyword}</span>"""
                        
                        st.markdown(f"<div>{keyword_html}</div>", unsafe_allow_html=True)
                    
                    # Display section summaries if available
                    if detect_sections and "sections" in result and result["sections"]:
                        st.subheader("Section Summaries")
                        
                        for section_name, section_summary in result["sections"].items():
                            with st.expander(section_name):
                                st.write(section_summary)
                    
                    # Show export button
                    if st.button("Export Summary to Text File", key="pdf_export_btn"):
                        # Create a comprehensive export with all information
                        export_content = f"# Summary of {uploaded_file.name}\n\n"
                        export_content += f"## Overall Summary\n{result['summary']}\n\n"
                        
                        # Add keywords if available
                        if extract_keywords and "keywords" in result and result["keywords"]:
                            export_content += "## Key Topics\n"
                            keywords_text = ", ".join([kw["keyword"] for kw in result["keywords"][:10]])
                            export_content += f"{keywords_text}\n\n"
                        
                        # Add section summaries if available
                        if detect_sections and "sections" in result and result["sections"]:
                            export_content += "## Section Summaries\n\n"
                            for section_name, section_summary in result["sections"].items():
                                export_content += f"### {section_name}\n{section_summary}\n\n"
                        
                        export_path = export_summary(export_content, uploaded_file.name)
                        st.success(f"Summary exported to: {export_path}")
                    
                    # Show rating widget
                    st.subheader("Rate this summary")
                    rating = st.slider("How would you rate the quality of this summary?", 
                                      1, 5, 3, 1, key="pdf_rating_slider")
                    
                    if st.button("Submit Rating", key="pdf_rating_btn"):
                        # Save rating (in a real app, you would store this in a database)
                        st.session_state["last_rating"] = rating
                        st.success(f"Thank you for rating this summary ({rating}/5)!")
                        
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
    else:
        st.info("Please upload a PDF file to generate a summary.")

def text_summarization_tab():
    """Handle text input and summarization"""
    st.header("Text Summarization")
    
    # Text input widget
    text = st.text_area("Enter text to summarize", height=300)
    
    if text:
        # Add options for keyword extraction and section detection
        extract_keywords = st.checkbox("Extract Keywords", value=True, 
                                      help="Extract key topics from the document",
                                      key="text_extract_keywords")
        detect_sections = st.checkbox("Detect & Summarize Sections", value=False,
                                     help="Identify document sections and summarize each separately",
                                     key="text_detect_sections")
        
        # Process button
        if st.button("Generate Summary", key="text_generate_btn"):
            # Initialize agent
            agent = initialize_agent()
            
            # Process text
            with st.spinner("Analyzing text and generating summary..."):
                try:
                    # Get summary with options
                    result = agent.summarize_text(
                        text, 
                        summary_length="normal",  # Default to normal length
                        extract_keywords=extract_keywords,
                        detect_sections=detect_sections
                    )
                    
                    # Store the analysis in session state for display
                    if "last_analysis" not in st.session_state:
                        st.session_state["last_analysis"] = {}
                    
                    # Display results
                    st.subheader("Summary")
                    st.write(result["summary"])
                    
                    # Display repetition metrics if available
                    if "repetition_metrics" in result:
                        with st.expander("Summary Quality Metrics"):
                            metrics = result["repetition_metrics"]
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Lower is better for repetition score (0-1)
                                quality_score = 100 - (metrics.get("repetition_score", 0) * 100)
                                st.metric("Quality Score", f"{quality_score:.1f}%")
                                
                            with col2:
                                # Higher is better for unique word ratio (0-1)
                                unique_ratio = metrics.get("unique_word_ratio", 0) * 100
                                st.metric("Unique Words", f"{unique_ratio:.1f}%")
                                
                            with col3:
                                # Lower is better for repeated phrase ratio (0-1)
                                repetition = metrics.get("repeated_phrase_ratio", 0) * 100
                                st.metric("Repetition", f"{repetition:.1f}%", 
                                         delta=f"-{repetition:.1f}%", delta_color="inverse")
                    
                    # Display keywords if available
                    if extract_keywords and "keywords" in result and result["keywords"]:
                        st.subheader("Key Topics")
                        
                        # Create clickable keyword tags with colors
                        keyword_html = ""
                        
                        for kw in result["keywords"][:10]:
                            keyword = kw["keyword"]
                            score = kw["score"]
                            # Higher score = more intense color
                            intensity = min(100, int(score * 100)) if score <= 1 else 100
                            keyword_html += f"""<span style="display: inline-block; 
                                                margin: 2px; padding: 4px 8px; 
                                                background-color: rgba(59, 130, 246, {intensity/100}); 
                                                border-radius: 16px; color: white; 
                                                font-size: 0.9em;">{keyword}</span>"""
                        
                        st.markdown(f"<div>{keyword_html}</div>", unsafe_allow_html=True)
                    
                    # Display section summaries if available
                    if detect_sections and "sections" in result and result["sections"]:
                        st.subheader("Section Summaries")
                        
                        for section_name, section_summary in result["sections"].items():
                            with st.expander(section_name):
                                st.write(section_summary)
                    
                    # Show export button
                    if st.button("Export Summary to Text File", key="text_export_btn"):
                        # Create a comprehensive export with all information
                        export_content = f"# Text Summary\n\n"
                        export_content += f"## Overall Summary\n{result['summary']}\n\n"
                        
                        # Add keywords if available
                        if extract_keywords and "keywords" in result and result["keywords"]:
                            export_content += "## Key Topics\n"
                            keywords_text = ", ".join([kw["keyword"] for kw in result["keywords"][:10]])
                            export_content += f"{keywords_text}\n\n"
                        
                        # Add section summaries if available
                        if detect_sections and "sections" in result and result["sections"]:
                            export_content += "## Section Summaries\n\n"
                            for section_name, section_summary in result["sections"].items():
                                export_content += f"### {section_name}\n{section_summary}\n\n"
                        
                        export_path = export_summary(export_content, "text_summary")
                        st.success(f"Summary exported to: {export_path}")
                    
                    # Show rating widget
                    st.subheader("Rate this summary")
                    rating = st.slider("How would you rate the quality of this summary?", 
                                      1, 5, 3, 1, key="text_rating_slider")
                    
                    if st.button("Submit Rating", key="text_rating_btn"):
                        # Save rating (in a real app, you would store this in a database)
                        st.session_state["last_rating"] = rating
                        st.success(f"Thank you for rating this summary ({rating}/5)!")
                        
                except Exception as e:
                    st.error(f"Error processing text: {str(e)}")
    else:
        st.info("Please enter some text to generate a summary.")

def main():
    """Main function to run the Streamlit app"""
    # Set page config
    st.set_page_config(
        page_title="Smart Notes Summarizer",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Display header
    display_header()
    
    # Display sidebar and get options
    sidebar_options = display_sidebar()
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["PDF Summarization", "Text Summarization"])
    
    # Handle each tab
    with tab1:
        pdf_summarization_tab()
    
    with tab2:
        text_summarization_tab()
    
    # Footer
    st.markdown("---")
    st.caption("Smart Notes Summarizer - Developed by Sanskriti Pal, IIT Goa")


if __name__ == "__main__":
    main()