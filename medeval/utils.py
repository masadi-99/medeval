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
    """Collect all JSON sample files from the samples directory, excluding checkpoints and other non-patient files"""
    sample_files = []
    
    for root, dirs, files in os.walk(samples_dir):
        # Skip .ipynb_checkpoints directories entirely
        dirs[:] = [d for d in dirs if d != '.ipynb_checkpoints']
        
        for file in files:
            if file.endswith('.json'):
                # Additional filtering to exclude checkpoint files and other non-patient files
                file_path = os.path.join(root, file)
                
                # Skip if file is in a checkpoint directory (extra safety)
                if '.ipynb_checkpoints' in file_path:
                    continue
                
                # Skip if filename suggests it's a checkpoint
                if 'checkpoint' in file.lower():
                    continue
                
                # Skip hidden files
                if file.startswith('.'):
                    continue
                
                sample_files.append(file_path)
    
    return sample_files


def load_sample(file_path: str) -> Dict:
    """Load a sample JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_flowchart_categories(flowchart_dir: str = None) -> List[str]:
    """Load all disease category names from flowchart files"""
    
    flowchart_directory = get_flowchart_directory(flowchart_dir)
    
    categories = []
    
    # Get category names from JSON filenames
    for filename in os.listdir(flowchart_directory):
        if filename.endswith('.json'):
            # Remove .json extension to get category name
            category = filename[:-5]
            categories.append(category)
    
    return sorted(categories)


def load_flowchart_content(category: str, flowchart_dir: str = None) -> Dict:
    """Load the content of a specific flowchart category"""
    
    flowchart_directory = get_flowchart_directory(flowchart_dir)
    filepath = os.path.join(flowchart_directory, f"{category}.json")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Flowchart file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_flowchart_for_prompt(category: str, flowchart_data: Dict) -> str:
    """Format flowchart data for inclusion in prompts"""
    
    formatted = f"**{category} Diagnostic Flowchart:**\n\n"
    
    def format_node(node, indent=0):
        """Recursively format flowchart nodes"""
        result = ""
        if isinstance(node, dict):
            for key, value in node.items():
                result += "  " * indent + f"• {key}\n"
                if isinstance(value, dict):
                    result += format_node(value, indent + 1)
                elif isinstance(value, list) and len(value) == 0:
                    result += "  " * (indent + 1) + f"→ Final diagnosis: {key}\n"
        return result
    
    # Format diagnostic tree
    if 'diagnostic' in flowchart_data:
        formatted += "Diagnostic Tree:\n"
        formatted += format_node(flowchart_data['diagnostic'])
    
    # Format knowledge base
    if 'knowledge' in flowchart_data:
        formatted += "\nClinical Knowledge:\n"
        knowledge = flowchart_data['knowledge']
        
        def format_knowledge(node, indent=0):
            result = ""
            if isinstance(node, dict):
                for key, value in node.items():
                    result += "  " * indent + f"• {key}:\n"
                    if isinstance(value, str):
                        result += "  " * (indent + 1) + f"{value}\n"
                    elif isinstance(value, dict):
                        result += format_knowledge(value, indent + 1)
            return result
        
        formatted += format_knowledge(knowledge)
    
    return formatted


def extract_diagnoses_from_flowchart(flowchart_data: Dict) -> List[str]:
    """Extract all possible diagnoses from a single flowchart"""
    
    diagnoses = []
    
    def extract_leaf_diagnoses(node):
        """Recursively extract leaf diagnoses from the flowchart structure"""
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, list) and len(value) == 0:  # leaf node (empty list)
                    diagnoses.append(key)
                elif isinstance(value, dict):
                    extract_leaf_diagnoses(value)
    
    # Extract from the diagnostic tree
    if 'diagnostic' in flowchart_data:
        extract_leaf_diagnoses(flowchart_data['diagnostic'])
    else:
        # Fallback: extract all leaf nodes from the entire structure
        extract_leaf_diagnoses(flowchart_data)
    
    return diagnoses


def get_flowchart_structure(flowchart_data: Dict) -> Dict:
    """Extract the flowchart structure for iterative reasoning"""
    
    if 'diagnostic' in flowchart_data:
        return flowchart_data['diagnostic']
    return flowchart_data


def get_flowchart_knowledge(flowchart_data: Dict) -> Dict:
    """Extract the knowledge base from flowchart data"""
    
    if 'knowledge' in flowchart_data:
        return flowchart_data['knowledge']
    return {}


def find_flowchart_root_nodes(flowchart_structure: Dict) -> List[str]:
    """Find the root nodes (starting points) in a flowchart"""
    
    if isinstance(flowchart_structure, dict):
        return list(flowchart_structure.keys())
    return []


def get_flowchart_children(flowchart_structure: Dict, node: str) -> List[str]:
    """Get the children nodes of a specific node in the flowchart"""
    
    def find_children(data, target_node):
        children = []
        if isinstance(data, dict):
            for key, value in data.items():
                if key == target_node:
                    if isinstance(value, dict):
                        children.extend(value.keys())
                    elif isinstance(value, list) and len(value) == 0:
                        # This is a leaf node
                        children = []
                    break
                elif isinstance(value, dict):
                    children.extend(find_children(value, target_node))
        return children
    
    return find_children(flowchart_structure, node)


def is_leaf_diagnosis(flowchart_structure: Dict, node: str) -> bool:
    """Check if a node is a leaf diagnosis (final diagnosis)"""
    
    def check_leaf(data, target_node):
        if isinstance(data, dict):
            for key, value in data.items():
                if key == target_node:
                    return isinstance(value, list) and len(value) == 0
                elif isinstance(value, dict):
                    result = check_leaf(value, target_node)
                    if result is not None:
                        return result
        return None
    
    result = check_leaf(flowchart_structure, node)
    return result if result is not None else False


def get_flowchart_first_step(flowchart_data: Dict) -> str:
    """Get the first step/starting point of a flowchart"""
    
    # Try to get from flowchart structure
    if 'diagnostic' in flowchart_data:
        structure = flowchart_data['diagnostic']
        if isinstance(structure, dict) and structure:
            # Get the first key (root node) from the diagnostic tree
            first_key = list(structure.keys())[0]
            
            # Create a more descriptive first step name
            if 'suspected' not in first_key.lower():
                if 'acute' in first_key.lower() or 'chronic' in first_key.lower():
                    return first_key  # Already descriptive
                else:
                    return f"Suspected {first_key}"
            else:
                return first_key
    
    # Fallback: use the category name from flowchart data
    if 'category' in flowchart_data:
        category = flowchart_data['category']
        return f"Suspected {category}"
    
    # Final fallback: generic starting point
    return "Initial Assessment"


def format_reasoning_step(step_num: int, current_node: str, available_options: List[str], 
                         knowledge: Dict, patient_data_summary: str) -> str:
    """Format a single reasoning step prompt"""
    
    prompt = f"**Reasoning Step {step_num}:**\n\n"
    prompt += f"Current diagnostic consideration: **{current_node}**\n\n"
    
    # Add patient data summary first
    prompt += f"**Patient Clinical Information:**\n{patient_data_summary}\n\n"
    
    # Add knowledge about current node
    if current_node in knowledge:
        knowledge_text = knowledge[current_node]
        if isinstance(knowledge_text, dict):
            prompt += f"**Clinical Criteria for {current_node}:**\n"
            for key, value in knowledge_text.items():
                prompt += f"• {key}: {value}\n"
        else:
            prompt += f"**Clinical Criteria for {current_node}:** {knowledge_text}\n"
        prompt += "\n"
    
    # Add available options with their clinical criteria
    if available_options:
        prompt += "**Available Next Diagnostic Considerations:**\n"
        for i, option in enumerate(available_options, 1):
            prompt += f"\n{i}. **{option}**\n"
            # Add knowledge for each option if available
            if option in knowledge:
                option_knowledge = knowledge[option]
                if isinstance(option_knowledge, dict):
                    prompt += f"   Clinical criteria:\n"
                    for key, value in option_knowledge.items():
                        # Provide FULL clinical criteria - no truncation!
                        # Evidence-based matching requires complete criteria
                        prompt += f"   • {key}: {value}\n"
                elif isinstance(option_knowledge, str):
                    # Provide FULL knowledge text - no truncation!
                    prompt += f"   Clinical criteria: {option_knowledge}\n"
            else:
                prompt += f"   (No specific clinical criteria available)\n"
        
        prompt += f"\n**CRITICAL INSTRUCTIONS:**\n"
        prompt += f"• Use ONLY the patient information and clinical criteria provided above\n"
        prompt += f"• Do NOT use general medical knowledge or assumptions\n"
        prompt += f"• Match SPECIFIC patient observations to SPECIFIC clinical criteria\n"
        prompt += f"• When choosing between options, explain why you selected one and rejected others\n"
        prompt += f"• Base your reasoning entirely on the available evidence\n\n"
        
        prompt += f"**Required Analysis Format:**\n\n"
        prompt += f"**EVIDENCE MATCHING:**\n"
        prompt += f"For each diagnostic option, identify:\n"
        prompt += f"• Which specific patient findings match the clinical criteria\n"
        prompt += f"• Which specific patient findings contradict the clinical criteria\n"
        prompt += f"• Missing information that would be expected for this diagnosis\n\n"
        
        prompt += f"**COMPARATIVE ANALYSIS:**\n"
        prompt += f"Compare the options based on evidence matching:\n"
        prompt += f"• Which option has the strongest evidence support?\n"
        prompt += f"• Which options can be ruled out and why?\n"
        prompt += f"• What specific evidence distinguishes your choice from alternatives?\n\n"
        
        prompt += f"**DECISION:** [Number of chosen option] - [Option name]\n"
        prompt += f"**RATIONALE:** [Specific evidence-based explanation for your choice and rejection of alternatives]\n"
    else:
        prompt += "This appears to be a final diagnosis. Based on the patient information and the diagnostic path taken, "
        prompt += "please confirm if this is the most appropriate primary discharge diagnosis.\n\n"
        
        prompt += f"**FINAL DIAGNOSIS VALIDATION:**\n"
        prompt += f"**EVIDENCE REVIEW:** Review all patient findings that support this final diagnosis\n"
        prompt += f"**CONFIRMATION:** Does the available evidence strongly support this diagnosis?\n"
        prompt += f"**DECISION:** CONFIRMED or RECONSIDER\n"
        prompt += f"**RATIONALE:** Evidence-based explanation for your final decision\n"
    
    return prompt


def extract_reasoning_choice(response: str, available_options: List[str]) -> Dict:
    """Extract the chosen option and reasoning from LLM response"""
    
    response_text = response.strip()
    
    # Try to extract structured response
    result = {
        'chosen_option': "",
        'evidence_matching': "",
        'comparative_analysis': "",
        'rationale': "",
        'decision_text': ""
    }
    
    # Extract sections
    import re
    
    # Extract EVIDENCE MATCHING section
    evidence_match = re.search(r'EVIDENCE MATCHING:\s*(.*?)(?=COMPARATIVE ANALYSIS:|DECISION:|$)', 
                              response_text, re.DOTALL | re.IGNORECASE)
    if evidence_match:
        result['evidence_matching'] = evidence_match.group(1).strip()
    
    # Extract COMPARATIVE ANALYSIS section
    comparative_match = re.search(r'COMPARATIVE ANALYSIS:\s*(.*?)(?=DECISION:|$)', 
                                 response_text, re.DOTALL | re.IGNORECASE)
    if comparative_match:
        result['comparative_analysis'] = comparative_match.group(1).strip()
    
    # Extract DECISION section
    decision_match = re.search(r'DECISION:\s*(.*?)(?=RATIONALE:|$)', 
                              response_text, re.DOTALL | re.IGNORECASE)
    if decision_match:
        result['decision_text'] = decision_match.group(1).strip()
    
    # Extract RATIONALE section
    rationale_match = re.search(r'RATIONALE:\s*(.*?)$', 
                               response_text, re.DOTALL | re.IGNORECASE)
    if rationale_match:
        result['rationale'] = rationale_match.group(1).strip()
    
    # For final diagnosis validation, also check for EVIDENCE REVIEW
    evidence_review_match = re.search(r'EVIDENCE REVIEW:\s*(.*?)(?=CONFIRMATION:|DECISION:|$)', 
                                     response_text, re.DOTALL | re.IGNORECASE)
    if evidence_review_match:
        result['evidence_matching'] = evidence_review_match.group(1).strip()
    
    # Parse the decision to extract the chosen option
    decision_text = result['decision_text']
    
    # Try to extract number and option name
    number_match = re.search(r'\b(\d+)\b', decision_text)
    if number_match:
        try:
            choice_num = int(number_match.group(1))
            if 1 <= choice_num <= len(available_options):
                result['chosen_option'] = available_options[choice_num - 1]
            else:
                # If number is out of range, try text matching
                result['chosen_option'] = _match_option_by_text(decision_text, available_options)
        except ValueError:
            result['chosen_option'] = _match_option_by_text(decision_text, available_options)
    else:
        # Try direct text matching if no number found
        result['chosen_option'] = _match_option_by_text(decision_text, available_options)
    
    # If still no match, try the whole response
    if not result['chosen_option']:
        result['chosen_option'] = _match_option_by_text(response_text, available_options)
    
    # Default to first option if parsing completely fails
    if not result['chosen_option'] and available_options:
        result['chosen_option'] = available_options[0]
    
    return result


def _match_option_by_text(text: str, available_options: List[str]) -> str:
    """Helper function to match option by text content"""
    text_lower = text.lower()
    
    # Direct matching
    for option in available_options:
        if option.lower() in text_lower:
            return option
    
    # Partial matching
    for option in available_options:
        option_words = option.lower().split()
        if any(word in text_lower for word in option_words if len(word) > 3):
            return option
    
    return "" 