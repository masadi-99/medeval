"""
Utility functions for MedEval package
"""

import json
import os
import pkg_resources
from typing import List


def get_data_path(filename: str = "") -> str:
    """Get path to data files within the package"""
    try:
        # Try to get from package data first
        if filename:
            return pkg_resources.resource_filename('medeval', f'data/{filename}')
        else:
            return pkg_resources.resource_filename('medeval', 'data')
    except:
        # Fallback to relative path (for development)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        data_path = os.path.join(parent_dir, 'data', filename)
        
        # If that doesn't work, try looking in the current working directory
        if not os.path.exists(data_path) and filename:
            cwd_path = os.path.join(os.getcwd(), 'data', filename)
            if os.path.exists(cwd_path):
                return cwd_path
        elif not os.path.exists(data_path) and not filename:
            cwd_path = os.path.join(os.getcwd(), 'data')
            if os.path.exists(cwd_path):
                return cwd_path
        
        return data_path


def load_possible_diagnoses(flowchart_dir: str = None) -> List[str]:
    """
    Load all possible primary discharge diagnoses from flowcharts
    
    Args:
        flowchart_dir: Directory containing diagnostic flowcharts. 
                      If None, uses package data.
    
    Returns:
        List of possible diagnoses
    """
    if flowchart_dir is None:
        flowchart_dir = get_data_path('Diagnosis_flowchart')
    
    if not os.path.exists(flowchart_dir):
        raise FileNotFoundError(f"Flowchart directory not found: {flowchart_dir}")
    
    diagnoses = []
    
    for filename in os.listdir(flowchart_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(flowchart_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            def extract_leaf_diagnoses(node):
                if isinstance(node, dict):
                    for key, value in node.items():
                        if isinstance(value, list) and len(value) == 0:  # leaf node
                            diagnoses.append(key)
                        elif isinstance(value, dict):
                            extract_leaf_diagnoses(value)
            
            extract_leaf_diagnoses(data['diagnostic'])
    
    return sorted(list(set(diagnoses)))


def extract_diagnosis_from_path(sample_path: str) -> str:
    """
    Extract ground truth diagnosis from sample file path
    
    Args:
        sample_path: Path to the sample JSON file
        
    Returns:
        Diagnosis name or None if not found
    """
    path_parts = sample_path.split('/')
    if len(path_parts) >= 3:
        return path_parts[-2]  # The diagnosis subdirectory
    return None


def get_samples_directory(samples_dir: str = None) -> str:
    """
    Get the samples directory path
    
    Args:
        samples_dir: Custom samples directory. If None, uses package data.
        
    Returns:
        Path to samples directory
    """
    if samples_dir is None:
        samples_dir = get_data_path('Finished')
    
    if not os.path.exists(samples_dir):
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    
    return samples_dir


def load_sample(sample_path: str) -> dict:
    """
    Load a sample JSON file
    
    Args:
        sample_path: Path to the JSON file
        
    Returns:
        Loaded JSON data
    """
    with open(sample_path, 'r') as f:
        return json.load(f)


def collect_sample_files(samples_dir: str) -> List[str]:
    """
    Collect all sample JSON files from the samples directory
    
    Args:
        samples_dir: Directory containing samples
        
    Returns:
        List of sample file paths
    """
    sample_files = []
    for root, dirs, files in os.walk(samples_dir):
        for file in files:
            if file.endswith('.json'):
                sample_files.append(os.path.join(root, file))
    
    return sample_files 