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
    extract_disease_category_from_path,
    get_samples_directory,
    load_sample,
    collect_sample_files,
    load_flowchart_categories,
    load_flowchart_content,
    format_flowchart_for_prompt,
    extract_diagnoses_from_flowchart
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
        self.flowchart_categories = load_flowchart_categories(flowchart_dir)
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
                       provide_diagnosis_list: bool, two_step_reasoning: bool = False,
                       num_categories: int = 3) -> Dict:
        """
        Evaluate a single sample
        
        Args:
            sample_path: Path to the sample file
            num_inputs: Number of input fields to use
            provide_diagnosis_list: Whether to provide diagnosis list
            two_step_reasoning: Whether to use two-step reasoning
            num_categories: Number of categories to select in first step
        
        Returns:
            Dict with evaluation results
        """
        sample = load_sample(sample_path)
        ground_truth = extract_diagnosis_from_path(sample_path)
        disease_category = extract_disease_category_from_path(sample_path)
        
        if two_step_reasoning:
            # Step 1: Category selection
            category_prompt = self.create_category_selection_prompt(sample, num_inputs, num_categories)
            category_response = self.query_llm(category_prompt)
            selected_categories = self.parse_selected_categories(category_response, num_categories)
            
            # Evaluate category selection accuracy
            category_correct = self.evaluate_category_selection(selected_categories, disease_category)
            
            if self.show_responses:
                print(f"Sample: {sample_path}")
                print(f"Disease Category: {disease_category}")
                print(f"Category Selection Response: '{category_response}'")
                print(f"Selected Categories: {selected_categories}")
                print(f"Category Selection Correct: {category_correct}")
                print()
            
            # Step 2: Final diagnosis with flowcharts
            final_prompt = self.create_two_step_final_prompt(sample, num_inputs, selected_categories)
            predicted = self.query_llm(final_prompt)
            
            if self.show_responses:
                print(f"Final Diagnosis Response: '{predicted}'")
                print(f"Ground Truth Diagnosis: '{ground_truth}'")
        
        else:
            # Standard single-step evaluation
            prompt = self.create_prompt(sample, num_inputs, provide_diagnosis_list)
            predicted = self.query_llm(prompt)
            
            # No category selection in single-step mode
            selected_categories = []
            category_response = ""
            category_correct = None
            
            if self.show_responses:
                print(f"Sample: {sample_path}")
                print(f"Disease Category: {disease_category}")
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
        
        result = {
            'sample_path': sample_path,
            'ground_truth': ground_truth,
            'disease_category': disease_category,
            'predicted_raw': predicted,
            'predicted_matched': matched_prediction,
            'correct': correct,
            'evaluation_method': 'llm_judge' if self.use_llm_judge else 'exact_match'
        }
        
        # Add two-step specific information
        if two_step_reasoning:
            result.update({
                'two_step_reasoning': True,
                'category_selection_response': category_response,
                'selected_categories': selected_categories,
                'category_selection_correct': category_correct,
                'category_prompt': category_prompt,
                'final_prompt': final_prompt
            })
        else:
            result['prompt'] = prompt
        
        return result
    
    def calculate_category_metrics(self, results: List[Dict]) -> Dict:
        """Calculate metrics for each disease category"""
        category_metrics = {}
        
        # Group results by disease category
        category_results = {}
        for result in results:
            category = result['disease_category']
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)
        
        # Calculate metrics for each category
        for category, cat_results in category_results.items():
            y_true = [r['ground_truth'] for r in cat_results]
            y_pred = [r['predicted_matched'] for r in cat_results]
            correct_predictions = [r['correct'] for r in cat_results]
            
            # Calculate accuracy directly from correct predictions
            accuracy = sum(correct_predictions) / len(correct_predictions) if correct_predictions else 0
            
            # For precision/recall/F1, use sklearn with unique labels from this category
            unique_labels = list(set(y_true + y_pred))
            
            if len(unique_labels) > 1:
                precision = precision_score(y_true, y_pred, labels=unique_labels, 
                                          average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, labels=unique_labels, 
                                    average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, labels=unique_labels, 
                             average='weighted', zero_division=0)
            else:
                # Single label case
                precision = accuracy
                recall = accuracy  
                f1 = accuracy
            
            category_metrics[category] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_samples': len(cat_results),
                'unique_diagnoses': len(set(y_true))
            }
        
        return category_metrics
    
    def evaluate_dataset(self, num_inputs: int = 6,
                        provide_diagnosis_list: bool = True, 
                        max_samples: Optional[int] = None,
                        samples_dir: str = None,
                        two_step_reasoning: bool = False,
                        num_categories: int = 3) -> Dict:
        """
        Evaluate the entire dataset
        
        Args:
            num_inputs: Number of input fields to use (1-6)
            provide_diagnosis_list: Whether to provide diagnosis list to LLM
            max_samples: Maximum number of samples to evaluate (None for all)
            samples_dir: Custom samples directory (optional)
            two_step_reasoning: Whether to use two-step reasoning
            num_categories: Number of categories to select in first step
        
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
        if two_step_reasoning:
            print(f"Two-step reasoning enabled (selecting {num_categories} categories)")
            print(f"Available categories: {len(self.flowchart_categories)}")
        print(f"LLM Judge enabled: {self.use_llm_judge}")
        if self.show_responses:
            print(f"Showing LLM responses: {self.show_responses}")
        print()
        
        results = []
        for i, sample_path in enumerate(sample_files):
            # Update progress more frequently for better user experience
            if len(sample_files) <= 20:
                # For small datasets, show every sample
                print(f"Progress: {i+1}/{len(sample_files)}")
            elif i % max(1, len(sample_files) // 10) == 0:
                # For larger datasets, show 10 progress updates
                print(f"Progress: {i+1}/{len(sample_files)}")
            
            result = self.evaluate_sample(
                sample_path, 
                num_inputs, 
                provide_diagnosis_list,
                two_step_reasoning=two_step_reasoning,
                num_categories=num_categories
            )
            results.append(result)
        
        print(f"Progress: {len(sample_files)}/{len(sample_files)} - Complete!")
        print()
        
        # Calculate overall metrics
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
        
        # Calculate category selection accuracy for two-step mode
        overall_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': len(results)
        }
        
        if two_step_reasoning:
            category_selections = [r.get('category_selection_correct') for r in results 
                                 if r.get('category_selection_correct') is not None]
            if category_selections:
                category_selection_accuracy = sum(category_selections) / len(category_selections)
                overall_metrics['category_selection_accuracy'] = category_selection_accuracy
        
        # Per-class metrics (individual diagnoses)
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
        
        # Calculate disease category metrics
        category_metrics = self.calculate_category_metrics(results)
        
        return {
            'overall_metrics': overall_metrics,
            'per_class_metrics': per_class_metrics,
            'category_metrics': category_metrics,
            'detailed_results': results,
            'configuration': {
                'num_inputs': num_inputs,
                'provide_diagnosis_list': provide_diagnosis_list,
                'model': self.model,
                'use_llm_judge': self.use_llm_judge,
                'show_responses': self.show_responses,
                'two_step_reasoning': two_step_reasoning,
                'num_categories': num_categories if two_step_reasoning else None
            }
        }
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

    def create_category_selection_prompt(self, sample: Dict, num_inputs: int, num_categories: int) -> str:
        """
        Create the prompt for the first step: category selection
        
        Args:
            sample: The clinical sample data
            num_inputs: Number of input fields to include (1 to 6)
            num_categories: Number of categories to select
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
        
        prompt = "You are a medical expert tasked with narrowing down the most likely disease categories based on clinical information.\n\n"
        
        # Add clinical data
        for i in range(1, min(num_inputs + 1, 7)):
            input_key = f"input{i}"
            if input_key in sample:
                prompt += f"**{input_descriptions[i]}:**\n{sample[input_key]}\n\n"
        
        prompt += "**Available Disease Categories:**\n"
        for i, category in enumerate(self.flowchart_categories, 1):
            prompt += f"{i}. {category}\n"
        
        prompt += f"\nBased on the clinical information above, select the {num_categories} most likely disease categories. "
        prompt += f"Respond with ONLY the {num_categories} category names, one per line, in order of likelihood (most likely first).\n\n"
        prompt += "Selected Categories:"
        
        return prompt
    
    def create_two_step_final_prompt(self, sample: Dict, num_inputs: int, 
                                   selected_categories: List[str]) -> str:
        """
        Create the prompt for the second step: final diagnosis with flowcharts
        
        Args:
            sample: The clinical sample data
            num_inputs: Number of input fields to include (1 to 6)
            selected_categories: Categories selected in the first step
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
        
        prompt = "You are a medical expert tasked with providing a primary discharge diagnosis based on clinical information and diagnostic flowcharts.\n\n"
        
        # Add clinical data
        for i in range(1, min(num_inputs + 1, 7)):
            input_key = f"input{i}"
            if input_key in sample:
                prompt += f"**{input_descriptions[i]}:**\n{sample[input_key]}\n\n"
        
        # Add flowcharts for selected categories
        prompt += "**Relevant Diagnostic Flowcharts:**\n\n"
        for category in selected_categories:
            try:
                flowchart_data = load_flowchart_content(category, self.flowchart_dir)
                flowchart_text = format_flowchart_for_prompt(category, flowchart_data)
                prompt += flowchart_text + "\n"
            except Exception as e:
                print(f"Warning: Could not load flowchart for {category}: {e}")
        
        # Add possible diagnoses
        prompt += "**Possible Primary Discharge Diagnoses:**\n"
        for i, diagnosis in enumerate(self.possible_diagnoses, 1):
            prompt += f"{i}. {diagnosis}\n"
        
        prompt += "\nBased on the clinical information and the diagnostic flowcharts above, "
        prompt += "provide your primary discharge diagnosis by selecting from the possible diagnoses list. "
        prompt += "Respond with ONLY the exact diagnosis name from the list.\n\n"
        prompt += "Primary Discharge Diagnosis:"
        
        return prompt
    
    def parse_selected_categories(self, response: str, num_categories: int) -> List[str]:
        """Parse the LLM response to extract selected categories"""
        
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        selected = []
        
        for line in lines[:num_categories]:  # Take only the requested number
            # Clean up the line - remove numbers, bullets, etc.
            clean_line = re.sub(r'^\d+\.?\s*', '', line)  # Remove leading numbers
            clean_line = re.sub(r'^[•\-\*]\s*', '', clean_line)  # Remove bullets
            clean_line = clean_line.strip()
            
            # Find best match in flowchart categories
            for category in self.flowchart_categories:
                if clean_line.lower() == category.lower():
                    selected.append(category)
                    break
                elif clean_line.lower() in category.lower() or category.lower() in clean_line.lower():
                    selected.append(category)
                    break
        
        # If we couldn't parse enough categories, fill with most common ones
        while len(selected) < num_categories and len(selected) < len(self.flowchart_categories):
            for category in self.flowchart_categories:
                if category not in selected:
                    selected.append(category)
                    break
        
        return selected[:num_categories]
    
    def evaluate_category_selection(self, selected_categories: List[str], ground_truth_category: str) -> bool:
        """Check if the ground truth category is in the selected categories"""
        return ground_truth_category in selected_categories 