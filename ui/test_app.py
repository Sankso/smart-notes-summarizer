#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit app for testing UI changes
"""

import streamlit as st

def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(
        page_title="Smart Notes Summarizer",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("Smart Notes Summarizer")
    st.write("This is a test app to demonstrate the UI changes.")
    
    tabs = st.tabs(["PDF Summarization", "Text Summarization"])
    
    with tabs[0]:
        # PDF Summarization tab
        st.header("PDF Summarization")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        
        if uploaded_file:
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
            
            if st.button("Generate Summary", key="pdf_generate_btn"):
                st.info("This is a test app. The model is not actually loaded.")
                
                st.subheader("Summary")
                st.write("This is a placeholder for the summary text.")
                
                if extract_keywords:
                    st.subheader("Key Topics")
                    keyword_html = ""
                    keywords = [
                        {"keyword": "Example", "score": 0.8},
                        {"keyword": "Test", "score": 0.6},
                        {"keyword": "Keywords", "score": 0.9}
                    ]
                    
                    for kw in keywords:
                        keyword = kw["keyword"]
                        score = kw["score"]
                        intensity = min(100, int(score * 100)) if score <= 1 else 100
                        keyword_html += f"""<span style="display: inline-block; 
                                            margin: 2px; padding: 4px 8px; 
                                            background-color: rgba(59, 130, 246, {intensity/100}); 
                                            border-radius: 16px; color: white; 
                                            font-size: 0.9em;">{keyword}</span>"""
                    
                    st.markdown(f"<div>{keyword_html}</div>", unsafe_allow_html=True)
                
                if detect_sections:
                    st.subheader("Section Summaries")
                    sections = {
                        "Introduction": "This is a test introduction summary.",
                        "Method": "This is a test method summary.",
                        "Results": "This is a test results summary."
                    }
                    
                    for section_name, section_summary in sections.items():
                        with st.expander(section_name):
                            st.write(section_summary)
        else:
            st.info("Please upload a PDF file to generate a summary.")
    
    with tabs[1]:
        # Text Summarization tab
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
                st.info("This is a test app. The model is not actually loaded.")
                
                st.subheader("Summary")
                st.write("This is a placeholder for the text summary.")
                
                # Display keywords if requested
                if extract_keywords:
                    st.subheader("Key Topics")
                    
                    # Create clickable keyword tags with colors
                    keyword_html = ""
                    keywords = [
                        {"keyword": "Example", "score": 0.8},
                        {"keyword": "Test", "score": 0.6},
                        {"keyword": "Keywords", "score": 0.9}
                    ]
                    
                    for kw in keywords:
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
                
                # Display section summaries if requested
                if detect_sections:
                    st.subheader("Section Summaries")
                    sections = {
                        "Introduction": "This is a test introduction summary.",
                        "Method": "This is a test method summary.",
                        "Results": "This is a test results summary."
                    }
                    
                    for section_name, section_summary in sections.items():
                        with st.expander(section_name):
                            st.write(section_summary)
        else:
            st.info("Please enter some text to generate a summary.")

if __name__ == "__main__":
    main()