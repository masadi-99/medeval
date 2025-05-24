"""
Main evaluation class for LLM diagnostic reasoning
"""

import json
import openai
from typing import List, Dict, Optional
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .utils import (
    load_possible_diagnoses, 
    extract_diagnosis_from_path, 
    get_samples_directory,
    load_sample,
    collect_sample_files
)


class DiagnosticEvaluator:
    """
    Main class for evaluating LLMs on diagnostic reasoning tasks
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", 
                 flowchart_dir: str = None, samples_dir: str = None,
                 use_llm_judge: bool = True, show_responses: bool = False):
        """
        Initialize the diagnostic evaluator
        
        Args:
            api_key: OpenAI API key
            model: Model to use for evaluation
            flowchart_dir: Directory containing diagnostic flowcharts (optional)
            samples_dir: Directory containing samples (optional)
            use_llm_judge: Whether to use LLM as a judge for evaluation
            show_responses: Whether to print LLM responses during evaluation
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.flowchart_dir = flowchart_dir
        self.samples_dir = samples_dir
        self.possible_diagnoses = load_possible_diagnoses(flowchart_dir)
        self.use_llm_judge = use_llm_judge
        self.show_responses = show_responses
        
    def create_prompt(self, sample: Dict, num_inputs: int, 
                      provide_diagnosis_list: bool) -> str:
        """
        Create the prompt for the LLM
        
        Args:
            sample: The clinical sample data
            num_inputs: Number of input fields to include (1 to 6)
            provide_diagnosis_list: Whether to provide the list of possible diagnoses
        """
        
        # Map input numbers to their clinical meanings
        input_descriptions = {
            1: "Chief Complaint",
            2: "History of Present Illness", 
            3: "Past Medical History",
            4: "Family History",
            5: "Physical Examination",
            6: "Laboratory Results and Pertinent Findings"
        }
        
        prompt = "You are a medical expert tasked with providing a primary discharge diagnosis based on clinical information.\n\n"
        
        # Add clinical data
        for i in range(1, min(num_inputs + 1, 7)):
            input_key = f"input{i}"
            if input_key in sample:
                prompt += f"**{input_descriptions[i]}:**\n{sample[input_key]}\n\n"
        
        if provide_diagnosis_list:
            prompt += "**Possible Primary Discharge Diagnoses:**\n"
            for i, diagnosis in enumerate(self.possible_diagnoses, 1):
                prompt += f"{i}. {diagnosis}\n"
            prompt += "\n"
            prompt += "Please provide your diagnosis by selecting from the list above. "
            prompt += "Respond with ONLY the exact diagnosis name from the list.\n\n"
        else:
            prompt += "Please provide the most likely primary discharge diagnosis. "
            prompt += "Respond with ONLY the diagnosis name.\n\n"
        
        prompt += "Primary Discharge Diagnosis:"
        
        return prompt
    
    def query_llm(self, prompt: str) -> str:
        """Query the LLM with the given prompt"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical expert providing diagnostic assessments."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return ""
    
    def create_judge_prompt(self, predicted: str, ground_truth: str) -> str:
        """Create prompt for LLM judge to evaluate if prediction matches ground truth"""
        prompt = f"""You are a medical expert evaluating diagnostic accuracy. 

Your task is to determine if two medical diagnoses refer to the same condition, accounting for:
- Different wording or phrasing
- Medical synonyms and abbreviations
- Alternative names for the same condition
- Clinical equivalents

Predicted Diagnosis: "{predicted}"
Ground Truth Diagnosis: "{ground_truth}"

Question: Do these two diagnoses refer to the same medical condition?

Respond with ONLY "YES" if they refer to the same condition, or "NO" if they refer to different conditions.

Answer:"""
        return prompt
    
    def llm_judge_evaluation(self, predicted: str, ground_truth: str) -> bool:
        """Use LLM to judge if prediction matches ground truth"""
        if predicted.strip().lower() == ground_truth.strip().lower():
            return True
        
        judge_prompt = self.create_judge_prompt(predicted, ground_truth)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical expert judge evaluating diagnostic equivalence."},
                    {"role": "user", "content": judge_prompt}
                ],
                max_tokens=10,
                temperature=0.0
            )
            
            judge_response = response.choices[0].message.content.strip().upper()
            return judge_response == "YES"
            
        except Exception as e:
            print(f"Error in LLM judge evaluation: {e}")
            # Fallback to exact match
            return predicted.strip().lower() == ground_truth.strip().lower()
    
    def normalize_diagnosis(self, diagnosis: str) -> str:
        """Normalize diagnosis name for comparison"""
        # Remove extra whitespace, punctuation, and convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', diagnosis.lower().strip())
        normalized = ' '.join(normalized.split())
        return normalized
    
    def find_best_match(self, predicted: str) -> str:
        """Find the best matching diagnosis from the possible diagnoses"""
        predicted_norm = self.normalize_diagnosis(predicted)
        
        # Exact match first
        for diagnosis in self.possible_diagnoses:
            if self.normalize_diagnosis(diagnosis) == predicted_norm:
                return diagnosis
        
        # Partial match
        for diagnosis in self.possible_diagnoses:
            if predicted_norm in self.normalize_diagnosis(diagnosis) or \
               self.normalize_diagnosis(diagnosis) in predicted_norm:
                return diagnosis
        
        return predicted  # Return original if no match found
    
    def evaluate_sample(self, sample_path: str, num_inputs: int, 
                       provide_diagnosis_list: bool) -> Dict:
        """
        Evaluate a single sample
        
        Returns:
            Dict with evaluation results
        """
        sample = load_sample(sample_path)
        ground_truth = extract_diagnosis_from_path(sample_path)
        
        prompt = self.create_prompt(sample, num_inputs, provide_diagnosis_list)
        predicted = self.query_llm(prompt)
        
        if self.show_responses:
            print(f"Sample: {sample_path}")
            print(f"Raw LLM Response: '{predicted}'")
            print(f"Ground Truth: '{ground_truth}'")
        
        # Normalize and match prediction
        matched_prediction = self.find_best_match(predicted)
        
        # Determine correctness
        if self.use_llm_judge:
            correct = self.llm_judge_evaluation(matched_prediction, ground_truth)
            if self.show_responses:
                print(f"LLM Judge Decision: {'CORRECT' if correct else 'INCORRECT'}")
        else:
            correct = ground_truth == matched_prediction
            if self.show_responses:
                print(f"Exact Match: {'CORRECT' if correct else 'INCORRECT'}")
        
        if self.show_responses:
            print("-" * 50)
        
        return {
            'sample_path': sample_path,
            'ground_truth': ground_truth,
            'predicted_raw': predicted,
            'predicted_matched': matched_prediction,
            'correct': correct,
            'prompt': prompt,
            'evaluation_method': 'llm_judge' if self.use_llm_judge else 'exact_match'
        }
    
    def evaluate_dataset(self, num_inputs: int = 6,
                        provide_diagnosis_list: bool = True, 
                        max_samples: Optional[int] = None,
                        samples_dir: str = None) -> Dict:
        """
        Evaluate the entire dataset
        
        Args:
            num_inputs: Number of input fields to use (1-6)
            provide_diagnosis_list: Whether to provide diagnosis list to LLM
            max_samples: Maximum number of samples to evaluate (None for all)
            samples_dir: Custom samples directory (optional)
        
        Returns:
            Dict with evaluation metrics and results
        """
        
        # Get samples directory
        if samples_dir is None:
            samples_dir = get_samples_directory(self.samples_dir)
        
        # Collect all sample files
        sample_files = collect_sample_files(samples_dir)
        
        if max_samples:
            sample_files = sample_files[:max_samples]
        
        print(f"Evaluating {len(sample_files)} samples...")
        print(f"Using {num_inputs} input fields")
        print(f"Providing diagnosis list: {provide_diagnosis_list}")
        print(f"LLM Judge enabled: {self.use_llm_judge}")
        if self.show_responses:
            print(f"Showing LLM responses: {self.show_responses}")
        print()
        
        results = []
        for i, sample_path in enumerate(sample_files):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(sample_files)}")
            
            result = self.evaluate_sample(sample_path, num_inputs, provide_diagnosis_list)
            results.append(result)
        
        # Calculate metrics
        y_true = [r['ground_truth'] for r in results]
        y_pred = [r['predicted_matched'] for r in results]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # For multi-class precision/recall, we need to handle the case where
        # some predicted classes might not be in the true classes
        unique_labels = list(set(y_true + y_pred))
        
        precision = precision_score(y_true, y_pred, labels=unique_labels, 
                                  average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, labels=unique_labels, 
                            average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=unique_labels, 
                     average='weighted', zero_division=0)
        
        # Per-class metrics
        per_class_metrics = {}
        for label in set(y_true):
            label_true = [1 if gt == label else 0 for gt in y_true]
            label_pred = [1 if pred == label else 0 for pred in y_pred]
            
            if sum(label_true) > 0:  # Only if there are true instances
                per_class_metrics[label] = {
                    'precision': precision_score(label_true, label_pred, zero_division=0),
                    'recall': recall_score(label_true, label_pred, zero_division=0),
                    'f1': f1_score(label_true, label_pred, zero_division=0),
                    'support': sum(label_true)
                }
        
        return {
            'overall_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_samples': len(results)
            },
            'per_class_metrics': per_class_metrics,
            'detailed_results': results,
            'configuration': {
                'num_inputs': num_inputs,
                'provide_diagnosis_list': provide_diagnosis_list,
                'model': self.model,
                'use_llm_judge': self.use_llm_judge,
                'show_responses': self.show_responses
            }
        }
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}") 