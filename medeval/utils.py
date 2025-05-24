"""
Utility functions for MedEval package
"""

import json
import os
import pkg_resources
from typing import List, Dict


def get_data_path() -> str:
    """Get the path to the data directory, with fallbacks for different installation methods"""
    
    # Method 1: Try pkg_resources (works for installed packages)
    try:
        return pkg_resources.resource_filename('medeval', 'data')
    except:
        pass
    
    # Method 2: Try relative to this file (works for development)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'data')
        if os.path.exists(data_path):
            return data_path
    except:
        pass
    
    # Method 3: Try current working directory
    try:
        cwd_data = os.path.join(os.getcwd(), 'medeval', 'data')
        if os.path.exists(cwd_data):
            return cwd_data
    except:
        pass
    
    # Method 4: Try looking for data in common locations
    possible_paths = [
        './data',
        '../data',
        './medeval/data',
        '../medeval/data'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    raise FileNotFoundError("Could not find data directory. Make sure the data files are available.")


def get_samples_directory(custom_dir: str = None) -> str:
    """Get the samples directory path"""
    if custom_dir and os.path.exists(custom_dir):
        return custom_dir
    
    data_path = get_data_path()
    samples_dir = os.path.join(data_path, 'Finished')
    
    if not os.path.exists(samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    
    return samples_dir


def get_flowchart_directory(custom_dir: str = None) -> str:
    """Get the flowchart directory path"""
    if custom_dir and os.path.exists(custom_dir):
        return custom_dir
    
    data_path = get_data_path()
    flowchart_dir = os.path.join(data_path, 'Diagnosis_flowchart')
    
    if not os.path.exists(flowchart_dir):
        raise FileNotFoundError(f"Flowchart directory not found: {flowchart_dir}")
    
    return flowchart_dir


def load_possible_diagnoses(flowchart_dir: str = None) -> List[str]:
    """Load all possible diagnoses from flowchart files"""
    
    flowchart_directory = get_flowchart_directory(flowchart_dir)
    
    diagnoses = []
    
    # Load from all JSON files in the flowchart directory
    for filename in os.listdir(flowchart_directory):
        if filename.endswith('.json'):
            filepath = os.path.join(flowchart_directory, filename)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Extract diagnoses from the flowchart data
                def extract_leaf_diagnoses(node):
                    """Recursively extract leaf diagnoses from the flowchart structure"""
                    if isinstance(node, dict):
                        for key, value in node.items():
                            if isinstance(value, list) and len(value) == 0:  # leaf node (empty list)
                                diagnoses.append(key)
                            elif isinstance(value, dict):
                                extract_leaf_diagnoses(value)
                
                # Extract from the diagnostic tree
                if 'diagnostic' in data:
                    extract_leaf_diagnoses(data['diagnostic'])
                else:
                    # Fallback: extract all leaf nodes from the entire structure
                    extract_leaf_diagnoses(data)
                    
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
    
    # Remove duplicates and sort
    unique_diagnoses = sorted(list(set(diagnoses)))
    
    return unique_diagnoses


def extract_diagnosis_from_path(file_path: str) -> str:
    """Extract the diagnosis name from the file path"""
    # The diagnosis is typically the last directory name before the filename
    # e.g., .../Disease/Subtype/file.json -> "Subtype"
    path_parts = file_path.replace('\\', '/').split('/')
    
    # Find the diagnosis (second to last directory)
    if len(path_parts) >= 2:
        return path_parts[-2]  # Directory containing the file
    
    return "Unknown"


def extract_disease_category_from_path(file_path: str) -> str:
    """Extract the disease category (main disease type) from the file path"""
    # The disease category is the first level directory under "Finished"
    # e.g., .../Finished/HeartDisease/NSTEMI/file.json -> "HeartDisease"
    
    path_parts = file_path.replace('\\', '/').split('/')
    
    # Find "Finished" in the path and get the next directory
    try:
        finished_index = next(i for i, part in enumerate(path_parts) if part == 'Finished')
        if finished_index + 1 < len(path_parts):
            return path_parts[finished_index + 1]
    except StopIteration:
        pass
    
    # Fallback: try to find disease category from path structure
    if len(path_parts) >= 3:
        # Assume structure like: .../Category/Diagnosis/file.json
        return path_parts[-3]
    
    return "Unknown"


def collect_sample_files(samples_dir: str) -> List[str]:
    """Collect all JSON sample files from the samples directory"""
    sample_files = []
    
    for root, dirs, files in os.walk(samples_dir):
        for file in files:
            if file.endswith('.json'):
                sample_files.append(os.path.join(root, file))
    
    return sample_files


def load_sample(file_path: str) -> Dict:
    """Load a sample JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f) 