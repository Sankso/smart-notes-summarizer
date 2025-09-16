#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze evaluation files to identify redundant or outdated ones
that can be archived or deleted. This is a helper script for cleaning up
the evaluation directory.
"""

import os
import re
import json
import logging
import argparse
from datetime import datetime
from collections import defaultdict
from tabulate import tabulate  # Install with: pip install tabulate

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def analyze_files(directory="./evaluation/results", list_all=False):
    """
    Analyze evaluation files and identify potential redundancies
    
    Args:
        directory: Directory to analyze
        list_all: Whether to list all files or just potential redundancies
    
    Returns:
        List of redundant files that could be deleted
    """
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return []
    
    # Group files by type
    file_types = defaultdict(list)
    
    # Find all relevant files and group them
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            # Extract file type and timestamp
            match = re.search(r"^([a-zA-Z_]+)_(\d{8}_\d{6})\.(json|md)$", file)
            if match:
                file_type = match.group(1)
                timestamp = match.group(2)
                extension = match.group(3)
                
                # Get file size
                size = os.path.getsize(file_path)
                
                # Try to get more details from JSON files
                details = {}
                if extension == "json" and file_path.endswith(".json"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Extract relevant info based on file type
                            if file_type == "unified_evaluation":
                                if "overall_average" in data:
                                    details["metrics"] = "ROUGE + BLEU"
                                    if "default" in data["overall_average"]:
                                        rouge1 = data["overall_average"]["default"].get("rouge1_f1", "N/A")
                                        bleu1 = data["overall_average"]["default"].get("bleu1", "N/A")
                                        details["rouge1"] = f"{rouge1:.4f}" if isinstance(rouge1, (int, float)) else rouge1
                                        details["bleu1"] = f"{bleu1:.4f}" if isinstance(bleu1, (int, float)) else bleu1
                            elif file_type in ["comprehensive_evaluation", "bleu_scores"]:
                                if "average_metrics" in data:
                                    if "rouge1_f1" in data["average_metrics"]:
                                        details["rouge1"] = f"{data['average_metrics']['rouge1_f1']:.4f}"
                                    if "bleu" in data["average_metrics"]:
                                        details["bleu"] = f"{data['average_metrics']['bleu']:.4f}"
                    except Exception as e:
                        logger.debug(f"Could not parse JSON file {file}: {e}")
                
                # Get datetime object from timestamp for age calculation
                try:
                    timestamp_dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                    age_days = (datetime.now() - timestamp_dt).days
                except:
                    age_days = "?"
                
                file_types[file_type].append({
                    "filename": file,
                    "timestamp": timestamp,
                    "datetime": timestamp_dt if 'timestamp_dt' in locals() else None,
                    "extension": extension,
                    "size": size,
                    "size_mb": size / (1024 * 1024),
                    "age_days": age_days,
                    "path": file_path,
                    "details": details
                })
    
    # Sort files by timestamp (newest first) within each type
    for file_type in file_types:
        file_types[file_type].sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Identify redundant files (keeping latest 3 for each type)
    redundant_files = []
    
    # Prepare data for table output
    table_data = []
    headers = ["File Type", "Filename", "Size (MB)", "Age (days)", "Keep?", "Details"]
    
    for file_type, files in file_types.items():
        for i, file in enumerate(files):
            keep = i < 3  # Keep the 3 most recent files
            if not keep:
                redundant_files.append(file["path"])
            
            # Only add to table if we're listing all or it's redundant
            if list_all or not keep:
                # Format the details
                details_str = ""
                if "details" in file:
                    details = file["details"]
                    if "metrics" in details:
                        details_str += f"Metrics: {details['metrics']}"
                    if "rouge1" in details:
                        details_str += f", ROUGE-1: {details['rouge1']}"
                    if "bleu1" in details:
                        details_str += f", BLEU-1: {details['bleu1']}"
                    elif "bleu" in details:
                        details_str += f", BLEU: {details['bleu']}"
                
                table_data.append([
                    file_type,
                    file["filename"],
                    f"{file['size_mb']:.2f}",
                    file["age_days"],
                    "Yes" if keep else "No",
                    details_str
                ])
    
    # Print the table
    if table_data:
        print("\nEvaluation Files Analysis:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Print summary
        print(f"\nFound {len(redundant_files)} redundant files that could be deleted.")
        print(f"Total space that could be freed: {sum(os.path.getsize(f) for f in redundant_files) / (1024*1024):.2f} MB")
    else:
        print("No evaluation files found for analysis.")
    
    return redundant_files

def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation files for potential cleanup")
    parser.add_argument("--dir", type=str, default="./evaluation/results",
                        help="Directory containing evaluation results")
    parser.add_argument("--all", action="store_true",
                        help="List all files, not just redundant ones")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete the redundant files (USE WITH CAUTION)")
    
    args = parser.parse_args()
    
    # Normalize path
    directory = os.path.abspath(args.dir)
    
    # Analyze files
    redundant_files = analyze_files(directory=directory, list_all=args.all)
    
    # Delete if requested
    if args.delete and redundant_files:
        print("\nDeleting redundant files...")
        for file_path in redundant_files:
            try:
                os.remove(file_path)
                print(f"Deleted: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        print(f"Deleted {len(redundant_files)} redundant files.")

if __name__ == "__main__":
    main()