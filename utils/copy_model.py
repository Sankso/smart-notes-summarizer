#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper script to copy fine-tuned model files to the correct location.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def copy_model_files(source_dir, destination_dir=None):
    """
    Copy fine-tuned model files to the correct location.
    
    Args:
        source_dir (str): Directory containing fine-tuned model files
        destination_dir (str, optional): Destination directory. If None, uses default.
    """
    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir
    
    # Set default destination if not provided
    if destination_dir is None:
        destination_dir = os.path.join(project_root, "models", "lora_weights")
    
    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
        return False
    
    try:
        # Get list of files in source directory
        files = os.listdir(source_dir)
        
        if not files:
            print(f"Error: Source directory {source_dir} is empty.")
            return False
        
        # Copy each file
        for file in files:
            source_file = os.path.join(source_dir, file)
            dest_file = os.path.join(destination_dir, file)
            
            if os.path.isfile(source_file):
                shutil.copy2(source_file, dest_file)
                print(f"Copied: {file}")
        
        print(f"\nSuccessfully copied model files to {destination_dir}")
        return True
        
    except Exception as e:
        print(f"Error copying files: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Copy fine-tuned model files to the correct location")
    parser.add_argument("source_dir", help="Directory containing fine-tuned model files")
    parser.add_argument("--dest", help="Destination directory (optional)")
    
    args = parser.parse_args()
    
    # Copy files
    success = copy_model_files(args.source_dir, args.dest)
    
    if success:
        print("\nYou can now run the application with your fine-tuned model.")
    else:
        print("\nFailed to copy model files. Please check the errors above.")

if __name__ == "__main__":
    main()