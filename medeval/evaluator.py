"""
Main evaluation class for LLM diagnostic reasoning
"""

import json
import openai
import asyncio
import time
from typing import List, Dict, Optional
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataclasses import dataclass
from collections import deque

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
    extract_diagnoses_from_flowchart,
    get_flowchart_structure,
    get_flowchart_knowledge,
    find_flowchart_root_nodes,
    get_flowchart_children,
    is_leaf_diagnosis,
    format_reasoning_step,
    extract_reasoning_choice
)

from .models import (
    create_model_provider,
    PREDEFINED_MODELS,
    ModelResponse
)


@dataclass
class RateLimiter:
    """Rate limiter for OpenAI API calls"""
    requests_per_minute: int = 450  # Leave some buffer below the 500 limit
    
    def __init__(self, requests_per_minute: int = 450):
        self.requests_per_minute = requests_per_minute
        self.requests = deque()
    
    async def wait_if_needed(self):
        """Wait if we need to respect rate limits"""
        now = time.time()
        # Remove requests older than 1 minute
        while self.requests and now - self.requests[0] > 60:
            self.requests.popleft()
        
        # If we're at the limit, wait
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0]) + 0.1  # Small buffer
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                # Clean up old requests after sleeping
                now = time.time()
                while self.requests and now - self.requests[0] > 60:
                    self.requests.popleft()
        
        # Record this request
        self.requests.append(now)


class DiagnosticEvaluator:
    """
    Main class for evaluating LLMs on diagnostic reasoning tasks
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", 
                 flowchart_dir: str = None, samples_dir: str = None,
                 use_llm_judge: bool = True, show_responses: bool = False,
                 max_concurrent: int = 10, 
                 # New parameters for model provider system
                 provider: str = "auto", huggingface_token: str = None,
                 device: str = "auto", torch_dtype: str = "auto",
                 thinking_mode: bool = True):
        """
        Initialize the diagnostic evaluator
        
        Args:
            api_key: OpenAI API key (for OpenAI models)
            model: Model name or predefined model key
            flowchart_dir: Directory containing diagnostic flowcharts (optional)
            samples_dir: Directory containing samples (optional)
            use_llm_judge: Whether to use LLM as a judge for evaluation
            show_responses: Whether to print LLM responses during evaluation
            max_concurrent: Maximum number of concurrent API calls
            provider: Model provider ("auto", "openai", "huggingface", "huggingface_api")
            huggingface_token: HuggingFace API token (for HF API models)
            device: Device for local models ("auto", "cpu", "cuda", etc.)
            torch_dtype: Torch dtype for local models ("auto", "float16", "bfloat16")
            thinking_mode: Enable thinking mode for compatible models (like Qwen3)
        """
        self.flowchart_dir = flowchart_dir
        self.samples_dir = samples_dir
        self.possible_diagnoses = load_possible_diagnoses(flowchart_dir)
        self.flowchart_categories = load_flowchart_categories(flowchart_dir)
        self.use_llm_judge = use_llm_judge
        self.show_responses = show_responses
        self.max_concurrent = max_concurrent
        self.rate_limiter = RateLimiter()
        
        # Determine model configuration
        self.model_name = model
        self.provider_type = provider
        
        # Check if model is in predefined configurations
        if model in PREDEFINED_MODELS:
            model_config = PREDEFINED_MODELS[model].copy()
            self.provider_type = model_config.get('provider', provider)
            print(f"Using predefined model configuration: {model}")
            print(f"Provider: {self.provider_type}")
            if 'model_name' in model_config:
                print(f"Model: {model_config['model_name']}")
            if model_config.get('thinking_mode') is not None:
                thinking_mode = model_config['thinking_mode']
                print(f"Thinking mode: {thinking_mode}")
        else:
            model_config = {}
        
        # Auto-detect provider if not specified
        if self.provider_type == "auto":
            if model.startswith(('gpt-', 'o1-')):
                self.provider_type = "openai"
            elif "/" in model:  # HuggingFace model format
                self.provider_type = "huggingface"
            else:
                self.provider_type = "openai"  # Default fallback
        
        # Create model provider
        if self.provider_type == "openai":
            if not api_key:
                import os
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API key required for OpenAI models")
            
            self.model_provider = create_model_provider(
                "openai",
                api_key=api_key,
                model=model_config.get('model', model),
                rate_limiter=self.rate_limiter
            )
            
        elif self.provider_type == "huggingface":
            self.model_provider = create_model_provider(
                "huggingface",
                model_name=model_config.get('model_name', model),
                device=device,
                torch_dtype=torch_dtype,
                thinking_mode=thinking_mode,
                max_length=model_config.get('max_length', 32768)
            )
            
        elif self.provider_type == "huggingface_api":
            if not huggingface_token:
                import os
                huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
                if not huggingface_token:
                    raise ValueError("HuggingFace token required for HuggingFace API models")
            
            self.model_provider = create_model_provider(
                "huggingface_api",
                model_name=model_config.get('model_name', model),
                api_token=huggingface_token,
                thinking_mode=thinking_mode
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider_type}")
        
        # For backward compatibility, keep these references
        self.model = self.model_name
        
        print(f"Initialized DiagnosticEvaluator with {self.provider_type} provider")
        if hasattr(self.model_provider, 'thinking_mode'):
            print(f"Thinking mode: {self.model_provider.thinking_mode}")
        print(f"Concurrent requests supported: {self.model_provider.supports_concurrent()}")
        print(f"Found {len(self.possible_diagnoses)} possible diagnoses")
        print(f"Found {len(self.flowchart_categories)} disease categories")
        
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
    
    def query_llm(self, prompt: str, max_tokens: int = None) -> str:
        """Query the LLM with the given prompt"""
        # Default to generous token limit for diagnostic reasoning
        if max_tokens is None:
            max_tokens = 500  # Generous default for complex diagnostic responses
        
        response = self.model_provider.query(prompt, max_tokens=max_tokens)
        
        if response.success:
            # For models with thinking mode, show thinking content if available
            if response.thinking_content and self.show_responses:
                print(f"🧠 Model thinking: {response.thinking_content[:200]}...")
            return response.content
        else:
            print(f"Error querying LLM: {response.error}")
            return ""
    
    async def query_llm_async(self, prompt: str, request_id: str = None, max_tokens: int = None) -> Dict:
        """Async query the LLM with the given prompt"""
        # Default to generous token limit for diagnostic reasoning
        if max_tokens is None:
            max_tokens = 500  # Generous default for complex diagnostic responses
        
        response = await self.model_provider.query_async(prompt, request_id=request_id, max_tokens=max_tokens)
        
        result = {
            'request_id': request_id,
            'response': response.content if response.success else "",
            'success': response.success,
            'error': response.error
        }
        
        # Add thinking content if available
        if response.thinking_content:
            result['thinking_content'] = response.thinking_content
        
        return result
    
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
        
        response = self.model_provider.query(judge_prompt, max_tokens=10, temperature=0.0)
        
        if response.success:
            judge_response = response.content.strip().upper()
            return judge_response == "YES"
        else:
            print(f"Error in LLM judge evaluation: {response.error}")
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
                       num_categories: int = 3, iterative_reasoning: bool = False,
                       max_reasoning_steps: int = 5) -> Dict:
        """
        Evaluate a single sample
        
        Args:
            sample_path: Path to the sample file
            num_inputs: Number of input fields to use
            provide_diagnosis_list: Whether to provide diagnosis list
            two_step_reasoning: Whether to use two-step reasoning
            num_categories: Number of categories to select in first step
            iterative_reasoning: Whether to use iterative step-by-step reasoning
            max_reasoning_steps: Maximum number of reasoning steps for iterative mode
        
        Returns:
            Dict with evaluation results
        """
        sample = load_sample(sample_path)
        ground_truth = extract_diagnosis_from_path(sample_path)
        disease_category = extract_disease_category_from_path(sample_path)
        
        if iterative_reasoning:
            # Step 1: Category selection (same as two-step)
            category_prompt = self.create_category_selection_prompt(sample, num_inputs, num_categories)
            # Category selection with detailed reasoning needs generous token limit
            category_response = self.query_llm(category_prompt, max_tokens=800)
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
            
            # Step 2: Iterative reasoning through flowcharts
            reasoning_result = self.iterative_reasoning_with_flowcharts(
                sample, num_inputs, selected_categories, max_reasoning_steps
            )
            
            predicted = reasoning_result['final_diagnosis']
            reasoning_trace = reasoning_result['reasoning_trace']
            reasoning_steps = reasoning_result['reasoning_steps']
            reasoning_path_correct = self.evaluate_reasoning_path(reasoning_trace, disease_category)
            
            if self.show_responses:
                print(f"Iterative Reasoning Steps: {reasoning_steps}")
                
                # Show initial category selection reasoning if multiple categories were provided
                if len(selected_categories) > 1:
                    print(f"Initial Category Selection Reasoning:")
                    print(f"  Available categories: {selected_categories}")
                    print(f"  Selection response: {reasoning_result.get('initial_category_selection_response', '')[:200]}...")
                    print(f"  Chosen category: {reasoning_result.get('category_used')}")
                    print()
                
                print(f"Reasoning Path:")
                for step in reasoning_trace:
                    if step.get('action') == 'start':
                        print(f"  Step {step['step']}: Starting with {step.get('category')} -> {step.get('current_node')}")
                    elif step.get('action') == 'reasoning_step':
                        print(f"  Step {step['step']}: {step.get('current_node')} -> {step.get('chosen_option')}")
                        if step.get('evidence_matching'):
                            print(f"    Evidence Matching: {step['evidence_matching'][:120]}...")
                        if step.get('comparative_analysis'):
                            print(f"    Comparative Analysis: {step['comparative_analysis'][:120]}...")
                        if step.get('rationale'):
                            print(f"    Rationale: {step['rationale'][:120]}...")
                    elif step.get('action') == 'final_diagnosis':
                        print(f"  Step {step['step']}: Final diagnosis - {step.get('current_node')}")
                print(f"Final Diagnosis: '{predicted}'")
                print(f"Ground Truth Diagnosis: '{ground_truth}'")
                print(f"Reasoning Path Correct: {reasoning_path_correct}")
                print()
        
        elif two_step_reasoning:
            # Step 1: Category selection
            category_prompt = self.create_category_selection_prompt(sample, num_inputs, num_categories)
            # Category selection with detailed reasoning needs generous token limit
            category_response = self.query_llm(category_prompt, max_tokens=800)
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
            # Final diagnosis reasoning needs generous token limit  
            predicted = self.query_llm(final_prompt, max_tokens=600)
            
            if self.show_responses:
                print(f"Final Diagnosis Response: '{predicted}'")
                print(f"Ground Truth Diagnosis: '{ground_truth}'")
        
        else:
            # Standard single-step evaluation
            prompt = self.create_prompt(sample, num_inputs, provide_diagnosis_list)
            # Standard diagnostic responses need reasonable token limit
            predicted = self.query_llm(prompt, max_tokens=500)
            
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
        
        # Add mode-specific information
        if iterative_reasoning:
            result.update({
                'iterative_reasoning': True,
                'category_selection_response': category_response,
                'selected_categories': selected_categories,
                'category_selection_correct': category_correct,
                'reasoning_trace': reasoning_trace,
                'reasoning_steps': reasoning_steps,
                'reasoning_path_correct': reasoning_path_correct,
                'category_prompt': category_prompt,
                'initial_category_selection_response': reasoning_result.get('initial_category_selection_response', ''),
                'chosen_category': reasoning_result.get('category_used', '')
            })
        elif two_step_reasoning:
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
                        num_categories: int = 3,
                        iterative_reasoning: bool = False,
                        max_reasoning_steps: int = 5) -> Dict:
        """
        Evaluate the entire dataset
        
        Args:
            num_inputs: Number of input fields to use (1-6)
            provide_diagnosis_list: Whether to provide diagnosis list to LLM
            max_samples: Maximum number of samples to evaluate (None for all)
            samples_dir: Custom samples directory (optional)
            two_step_reasoning: Whether to use two-step reasoning
            num_categories: Number of categories to select in first step
            iterative_reasoning: Whether to use iterative step-by-step reasoning
            max_reasoning_steps: Maximum number of reasoning steps for iterative mode
        
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
        elif iterative_reasoning:
            print(f"Iterative reasoning enabled (selecting {num_categories} categories, max {max_reasoning_steps} steps)")
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
                num_categories=num_categories,
                iterative_reasoning=iterative_reasoning,
                max_reasoning_steps=max_reasoning_steps
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
        
        # Calculate mode-specific metrics
        overall_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': len(results)
        }
        
        if two_step_reasoning or iterative_reasoning:
            category_selections = [r.get('category_selection_correct') for r in results 
                                 if r.get('category_selection_correct') is not None]
            if category_selections:
                category_selection_accuracy = sum(category_selections) / len(category_selections)
                overall_metrics['category_selection_accuracy'] = category_selection_accuracy
        
        if iterative_reasoning:
            # Calculate reasoning path metrics
            reasoning_paths = [r.get('reasoning_path_correct') for r in results 
                             if r.get('reasoning_path_correct') is not None]
            if reasoning_paths:
                reasoning_path_accuracy = sum(reasoning_paths) / len(reasoning_paths)
                overall_metrics['reasoning_path_accuracy'] = reasoning_path_accuracy
            
            # Calculate average reasoning steps
            reasoning_steps = [r.get('reasoning_steps', 0) for r in results 
                             if r.get('reasoning_steps') is not None]
            if reasoning_steps:
                avg_reasoning_steps = sum(reasoning_steps) / len(reasoning_steps)
                overall_metrics['avg_reasoning_steps'] = avg_reasoning_steps
        
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
                'iterative_reasoning': iterative_reasoning,
                'num_categories': num_categories if (two_step_reasoning or iterative_reasoning) else None,
                'max_reasoning_steps': max_reasoning_steps if iterative_reasoning else None
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
        prompt += "**Patient Clinical Information:**\n"
        for i in range(1, min(num_inputs + 1, 7)):
            input_key = f"input{i}"
            if input_key in sample:
                prompt += f"• {input_descriptions[i]}: {sample[input_key]}\n"
        prompt += "\n"
        
        prompt += "**Available Disease Categories:**\n"
        for i, category in enumerate(self.flowchart_categories, 1):
            prompt += f"{i}. {category}\n"
        
        if num_categories > 1:
            prompt += f"\nBased on the patient clinical information above, you must select the {num_categories} most likely disease categories.\n\n"
            prompt += f"**IMPORTANT INSTRUCTIONS:**\n"
            prompt += f"• Use ONLY the specific clinical findings, symptoms, and data provided above\n"
            prompt += f"• Do NOT rely on general medical knowledge\n"
            prompt += f"• Match specific patient observations to each category's typical presentations\n"
            prompt += f"• Provide detailed reasoning for your choices and rejections\n\n"
            
            prompt += f"Please provide your analysis in this format:\n\n"
            prompt += f"**DETAILED ANALYSIS:**\n"
            prompt += f"For each category you are considering, analyze:\n"
            prompt += f"• Which specific patient findings support this category\n"
            prompt += f"• Which specific patient findings argue against this category\n"
            prompt += f"• Your reasoning based on the available clinical data\n\n"
            
            prompt += f"**SELECTED CATEGORIES:**\n"
            prompt += f"List your {num_categories} chosen categories in order of likelihood:\n"
            prompt += f"1. [Category Name] - [Brief justification based on patient data]\n"
            prompt += f"2. [Category Name] - [Brief justification based on patient data]\n"
            if num_categories > 2:
                prompt += f"3. [Category Name] - [Brief justification based on patient data]\n"
            if num_categories > 3:
                for i in range(4, num_categories + 1):
                    prompt += f"{i}. [Category Name] - [Brief justification based on patient data]\n"
            
            prompt += f"\n**REJECTED CATEGORIES:**\n"
            prompt += f"Explain why you rejected the most obvious alternative categories based on the patient data.\n"
        else:
            prompt += f"\nBased on the patient clinical information above, select the 1 most likely disease category.\n"
            prompt += f"Respond with ONLY the category name that best matches the patient's presentation.\n"
        
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
        
        selected = []
        
        # Try to extract from SELECTED CATEGORIES section first
        import re
        
        # Look for the SELECTED CATEGORIES section
        selected_section_match = re.search(r'SELECTED CATEGORIES:\s*(.*?)(?=REJECTED CATEGORIES:|$)', 
                                         response, re.DOTALL | re.IGNORECASE)
        
        if selected_section_match:
            selected_text = selected_section_match.group(1)
            
            # Extract numbered categories
            lines = [line.strip() for line in selected_text.split('\n') if line.strip()]
            
            for line in lines[:num_categories]:
                # Extract category name from lines like "1. Category Name - justification"
                category_match = re.search(r'^\d+\.\s*([^-]+)', line)
                if category_match:
                    category_name = category_match.group(1).strip()
                    
                    # Find best match in flowchart categories
                    for category in self.flowchart_categories:
                        if category_name.lower() == category.lower():
                            selected.append(category)
                            break
                        elif category_name.lower() in category.lower() or category.lower() in category_name.lower():
                            selected.append(category)
                            break
        
        # Fallback to original parsing if structured parsing failed
        if len(selected) < num_categories:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            for line in lines[:num_categories]:
                # Clean up the line - remove numbers, bullets, etc.
                clean_line = re.sub(r'^\d+\.?\s*', '', line)  # Remove leading numbers
                clean_line = re.sub(r'^[•\-\*]\s*', '', clean_line)  # Remove bullets
                clean_line = clean_line.split('-')[0].strip()  # Remove everything after dash
                clean_line = clean_line.strip()
                
                # Find best match in flowchart categories
                for category in self.flowchart_categories:
                    if clean_line.lower() == category.lower():
                        if category not in selected:
                            selected.append(category)
                        break
                    elif clean_line.lower() in category.lower() or category.lower() in clean_line.lower():
                        if category not in selected:
                            selected.append(category)
                        break
        
        # If we still couldn't parse enough categories, fill with most common ones
        while len(selected) < num_categories and len(selected) < len(self.flowchart_categories):
            for category in self.flowchart_categories:
                if category not in selected:
                    selected.append(category)
                    break
        
        return selected[:num_categories]
    
    def evaluate_category_selection(self, selected_categories: List[str], ground_truth_category: str) -> bool:
        """Check if the ground truth category is in the selected categories"""
        return ground_truth_category in selected_categories
    
    def create_patient_data_summary(self, sample: Dict, num_inputs: int) -> str:
        """Create a complete summary of patient data for reasoning steps"""
        
        input_descriptions = {
            1: "Chief Complaint",
            2: "History of Present Illness", 
            3: "Past Medical History",
            4: "Family History",
            5: "Physical Examination",
            6: "Laboratory Results and Pertinent Findings"
        }
        
        summary = ""
        for i in range(1, min(num_inputs + 1, 7)):
            input_key = f"input{i}"
            if input_key in sample:
                # Provide FULL clinical information - no truncation!
                # Evidence-based reasoning requires complete patient data
                content = sample[input_key]
                summary += f"• {input_descriptions[i]}: {content}\n"
        
        return summary.strip()
    
    def iterative_reasoning_with_flowcharts(self, sample: Dict, num_inputs: int, 
                                          selected_categories: List[str], 
                                          max_steps: int = 5) -> Dict:
        """
        Perform iterative step-by-step reasoning following flowcharts
        
        Args:
            sample: The clinical sample data
            num_inputs: Number of input fields to use
            selected_categories: Categories selected in the first step
            max_steps: Maximum number of reasoning steps
        
        Returns:
            Dict with reasoning results
        """
        
        patient_summary = self.create_patient_data_summary(sample, num_inputs)
        reasoning_trace = []
        final_diagnosis = None
        
        # Load flowcharts for selected categories
        flowcharts = {}
        for category in selected_categories:
            try:
                flowchart_data = load_flowchart_content(category, self.flowchart_dir)
                flowcharts[category] = {
                    'structure': get_flowchart_structure(flowchart_data),
                    'knowledge': get_flowchart_knowledge(flowchart_data)
                }
            except Exception as e:
                print(f"Warning: Could not load flowchart for {category}: {e}")
        
        if not flowcharts:
            return {
                'final_diagnosis': "",
                'reasoning_trace': [],
                'reasoning_steps': 0,
                'reasoning_successful': False
            }
        
        # Step 1: Select starting category and root node
        initial_category_response = ""
        if len(selected_categories) == 1:
            current_category = selected_categories[0]
            initial_category_response = f"Single category selected: {current_category}"
        else:
            # Let LLM choose which category to start with using enhanced evidence-based reasoning
            category_prompt = self.create_initial_category_selection_prompt(
                patient_summary, selected_categories, flowcharts
            )
            initial_category_response = self.query_llm(category_prompt, max_tokens=800)
            current_category = self.parse_category_selection(initial_category_response, selected_categories)
        
        if current_category not in flowcharts:
            current_category = selected_categories[0]  # Fallback
        
        # Get root nodes for the selected category
        root_nodes = find_flowchart_root_nodes(flowcharts[current_category]['structure'])
        if not root_nodes:
            return {
                'final_diagnosis': "",
                'reasoning_trace': [],
                'reasoning_steps': 0,
                'reasoning_successful': False
            }
        
        current_node = root_nodes[0]  # Start with first root node
        
        reasoning_trace.append({
            'step': 1,
            'category': current_category,
            'current_node': current_node,
            'action': 'start',
            'response': f"Starting with {current_category} -> {current_node}"
        })
        
        # Iterative reasoning through the flowchart
        for step in range(2, max_steps + 1):
            
            # Check if current node is a leaf (final diagnosis)
            if is_leaf_diagnosis(flowcharts[current_category]['structure'], current_node):
                final_diagnosis = current_node
                reasoning_trace.append({
                    'step': step,
                    'category': current_category,
                    'current_node': current_node,
                    'action': 'final_diagnosis',
                    'response': f"Reached final diagnosis: {current_node}"
                })
                break
            
            # Get children nodes
            children = get_flowchart_children(flowcharts[current_category]['structure'], current_node)
            
            if not children:
                # This is effectively a leaf node
                final_diagnosis = current_node
                reasoning_trace.append({
                    'step': step,
                    'category': current_category,
                    'current_node': current_node,
                    'action': 'final_diagnosis',
                    'response': f"No further options, final diagnosis: {current_node}"
                })
                break
            
            # Create reasoning step prompt
            step_prompt = format_reasoning_step(
                step, 
                current_node, 
                children,
                flowcharts[current_category]['knowledge'],
                patient_summary
            )
            
            # Get LLM response
            # Iterative reasoning steps need large token limit for evidence matching and comparative analysis
            step_response = self.query_llm(step_prompt, max_tokens=1200)
            
            # Parse the choice and reasoning
            reasoning_result = extract_reasoning_choice(step_response, children)
            chosen_node = reasoning_result['chosen_option']
            
            reasoning_trace.append({
                'step': step,
                'category': current_category,
                'current_node': current_node,
                'available_options': children,
                'chosen_option': chosen_node,
                'action': 'reasoning_step',
                'prompt': step_prompt,
                'response': step_response,
                'evidence_matching': reasoning_result.get('evidence_matching', ''),
                'comparative_analysis': reasoning_result.get('comparative_analysis', ''),
                'rationale': reasoning_result.get('rationale', ''),
                'decision_text': reasoning_result.get('decision_text', '')
            })
            
            current_node = chosen_node
        
        # If we didn't reach a final diagnosis, use the last node
        if not final_diagnosis:
            final_diagnosis = current_node
        
        return {
            'final_diagnosis': final_diagnosis,
            'reasoning_trace': reasoning_trace,
            'reasoning_steps': len(reasoning_trace),
            'reasoning_successful': final_diagnosis != "",
            'category_used': current_category,
            'initial_category_selection_response': initial_category_response,
            'selected_categories': selected_categories
        }
    
    def create_initial_category_selection_prompt(self, patient_summary: str, 
                                               categories: List[str], 
                                               flowcharts: Dict) -> str:
        """Create prompt for selecting which category to start reasoning with (with detailed evidence-based reasoning)"""
        
        prompt = "You are a medical expert beginning iterative diagnostic reasoning. You have already identified the most likely disease categories, and now you must decide which category to explore FIRST in your step-by-step diagnostic process.\n\n"
        
        # Add patient data
        prompt += f"**Patient Clinical Information:**\n{patient_summary}\n\n"
        
        # Add the selected categories with brief descriptions if available
        prompt += f"**Previously Selected Disease Categories to Explore:**\n"
        for i, category in enumerate(categories, 1):
            prompt += f"{i}. **{category}**\n"
            # Add brief flowchart info if available
            if category in flowcharts and 'knowledge' in flowcharts[category]:
                # Add a brief overview of this category's typical presentations
                knowledge = flowcharts[category]['knowledge']
                if isinstance(knowledge, dict) and knowledge:
                    prompt += f"   Typical presentations: Focus on conditions related to {category.lower()}\n"
            prompt += "\n"
        
        # Instructions for evidence-based reasoning
        prompt += f"**CRITICAL INSTRUCTIONS:**\n"
        prompt += f"• Use ONLY the patient information provided above\n"
        prompt += f"• Match SPECIFIC patient observations to each category's typical presentations\n"
        prompt += f"• Choose the category with the STRONGEST evidence for immediate exploration\n"
        prompt += f"• Provide detailed reasoning for your choice and deferral of others\n\n"
        
        if len(categories) > 1:
            prompt += f"**Required Analysis Format:**\n\n"
            prompt += f"**EVIDENCE ANALYSIS:**\n"
            prompt += f"For each category, analyze:\n"
            prompt += f"• Which specific patient findings support starting with this category\n"
            prompt += f"• Which specific patient findings suggest this category should be deferred\n"
            prompt += f"• The urgency/priority of exploring this category first\n\n"
            
            prompt += f"**COMPARATIVE REASONING:**\n"
            prompt += f"Compare the categories for initial exploration:\n"
            prompt += f"• Which category has the most compelling evidence to start with?\n"
            prompt += f"• Why should the other {len(categories)-1} categories be explored later?\n"
            prompt += f"• What specific evidence makes your chosen category the highest priority?\n\n"
            
            prompt += f"**DECISION:** [Number] - [Category Name]\n"
            prompt += f"**RATIONALE:** [Detailed evidence-based explanation for starting with this category and deferring the other {len(categories)-1} categories]\n"
        else:
            # Single category case
            prompt += f"**Evidence Analysis:**\n"
            prompt += f"Analyze why this category is the appropriate starting point:\n"
            prompt += f"• Which specific patient findings support exploring this category\n"
            prompt += f"• Why is this category the logical place to begin diagnostic reasoning\n\n"
            
            prompt += f"**DECISION:** 1 - {categories[0]}\n"
            prompt += f"**RATIONALE:** [Evidence-based explanation for why this category is the appropriate starting point]\n"
        
        return prompt
    
    def parse_category_selection(self, response: str, categories: List[str]) -> str:
        """Parse enhanced category selection response with evidence-based reasoning"""
        
        import re
        
        # First try to extract from DECISION section
        decision_match = re.search(r'DECISION:\s*(\d+)\s*-\s*([^.\n]+)', response, re.IGNORECASE)
        if decision_match:
            try:
                choice_num = int(decision_match.group(1))
                if 1 <= choice_num <= len(categories):
                    return categories[choice_num - 1]
            except ValueError:
                pass
        
        # Fallback: look for any number in the response
        number_match = re.search(r'\b(\d+)\b', response.strip())
        if number_match:
            try:
                choice_num = int(number_match.group(1))
                if 1 <= choice_num <= len(categories):
                    return categories[choice_num - 1]
            except ValueError:
                pass
        
        # Fallback: try to match category names directly
        response_lower = response.lower()
        for category in categories:
            if category.lower() in response_lower:
                return category
        
        # Final fallback to first category
        return categories[0] if categories else ""
    
    def evaluate_reasoning_path(self, reasoning_trace: List[Dict], ground_truth_category: str) -> bool:
        """Evaluate if the reasoning path used the correct disease category"""
        
        if not reasoning_trace:
            return False
        
        # Check if any step used the correct category
        for step in reasoning_trace:
            if step.get('category') == ground_truth_category:
                return True
        
        return False
    
    async def iterative_reasoning_with_flowcharts_async(self, sample: Dict, num_inputs: int, 
                                                      selected_categories: List[str], 
                                                      max_steps: int = 5) -> Dict:
        """
        Async version of iterative reasoning that batches API calls for speed
        """
        
        patient_summary = self.create_patient_data_summary(sample, num_inputs)
        reasoning_trace = []
        final_diagnosis = None
        
        # Load flowcharts for selected categories
        flowcharts = {}
        for category in selected_categories:
            try:
                flowchart_data = load_flowchart_content(category, self.flowchart_dir)
                flowcharts[category] = {
                    'structure': get_flowchart_structure(flowchart_data),
                    'knowledge': get_flowchart_knowledge(flowchart_data)
                }
            except Exception as e:
                print(f"Warning: Could not load flowchart for {category}: {e}")
        
        if not flowcharts:
            return {
                'final_diagnosis': "",
                'reasoning_trace': [],
                'reasoning_steps': 0,
                'reasoning_successful': False
            }
        
        # Step 1: Select starting category and root node
        initial_category_response = ""
        if len(selected_categories) == 1:
            current_category = selected_categories[0]
            initial_category_response = f"Single category selected: {current_category}"
        else:
            # Let LLM choose which category to start with using enhanced evidence-based reasoning
            category_prompt = self.create_initial_category_selection_prompt(
                patient_summary, selected_categories, flowcharts
            )
            initial_category_response = self.query_llm(category_prompt, max_tokens=800)
            current_category = self.parse_category_selection(initial_category_response, selected_categories)
        
        if current_category not in flowcharts:
            current_category = selected_categories[0]  # Fallback
        
        # Get root nodes for the selected category
        root_nodes = find_flowchart_root_nodes(flowcharts[current_category]['structure'])
        if not root_nodes:
            return {
                'final_diagnosis': "",
                'reasoning_trace': [],
                'reasoning_steps': 0,
                'reasoning_successful': False
            }
        
        current_node = root_nodes[0]  # Start with first root node
        
        reasoning_trace.append({
            'step': 1,
            'category': current_category,
            'current_node': current_node,
            'action': 'start',
            'response': f"Starting with {current_category} -> {current_node}"
        })
        
        # Iterative reasoning through the flowchart
        for step in range(2, max_steps + 1):
            
            # Check if current node is a leaf (final diagnosis)
            if is_leaf_diagnosis(flowcharts[current_category]['structure'], current_node):
                final_diagnosis = current_node
                reasoning_trace.append({
                    'step': step,
                    'category': current_category,
                    'current_node': current_node,
                    'action': 'final_diagnosis',
                    'response': f"Reached final diagnosis: {current_node}"
                })
                break
            
            # Get children nodes
            children = get_flowchart_children(flowcharts[current_category]['structure'], current_node)
            
            if not children:
                # This is effectively a leaf node
                final_diagnosis = current_node
                reasoning_trace.append({
                    'step': step,
                    'category': current_category,
                    'current_node': current_node,
                    'action': 'final_diagnosis',
                    'response': f"No further options, final diagnosis: {current_node}"
                })
                break
            
            # Create reasoning step prompt
            step_prompt = format_reasoning_step(
                step, 
                current_node, 
                children,
                flowcharts[current_category]['knowledge'],
                patient_summary
            )
            
            # Get LLM response
            # Iterative reasoning steps need large token limit for evidence matching and comparative analysis
            step_response = self.query_llm(step_prompt, max_tokens=1200)
            
            # Parse the choice and reasoning
            reasoning_result = extract_reasoning_choice(step_response, children)
            chosen_node = reasoning_result['chosen_option']
            
            reasoning_trace.append({
                'step': step,
                'category': current_category,
                'current_node': current_node,
                'available_options': children,
                'chosen_option': chosen_node,
                'action': 'reasoning_step',
                'prompt': step_prompt,
                'response': step_response,
                'evidence_matching': reasoning_result.get('evidence_matching', ''),
                'comparative_analysis': reasoning_result.get('comparative_analysis', ''),
                'rationale': reasoning_result.get('rationale', ''),
                'decision_text': reasoning_result.get('decision_text', '')
            })
            
            current_node = chosen_node
        
        # If we didn't reach a final diagnosis, use the last node
        if not final_diagnosis:
            final_diagnosis = current_node
        
        return {
            'final_diagnosis': final_diagnosis,
            'reasoning_trace': reasoning_trace,
            'reasoning_steps': len(reasoning_trace),
            'reasoning_successful': final_diagnosis != "",
            'category_used': current_category,
            'initial_category_selection_response': initial_category_response,
            'selected_categories': selected_categories
        }
    
    async def evaluate_sample_async(self, sample_path: str, num_inputs: int, 
                                   provide_diagnosis_list: bool, two_step_reasoning: bool = False,
                                   num_categories: int = 3, iterative_reasoning: bool = False,
                                   max_reasoning_steps: int = 5) -> Dict:
        """
        Async version of evaluate_sample for concurrent processing
        """
        sample = load_sample(sample_path)
        ground_truth = extract_diagnosis_from_path(sample_path)
        disease_category = extract_disease_category_from_path(sample_path)
        
        # Prepare API calls that need to be made
        api_calls = []
        
        if iterative_reasoning:
            # Step 1: Category selection
            category_prompt = self.create_category_selection_prompt(sample, num_inputs, num_categories)
            category_call = self.query_llm_async(category_prompt, f"category_{sample_path}", max_tokens=800)
            api_calls.append(('category_selection', category_call))
            
            # We'll do iterative reasoning after category selection
            
        elif two_step_reasoning:
            # Step 1: Category selection  
            category_prompt = self.create_category_selection_prompt(sample, num_inputs, num_categories)
            category_call = self.query_llm_async(category_prompt, f"category_{sample_path}", max_tokens=800)
            api_calls.append(('category_selection', category_call))
            
            # Step 2 will be done after category selection
            
        else:
            # Standard single-step evaluation
            prompt = self.create_prompt(sample, num_inputs, provide_diagnosis_list)
            main_call = self.query_llm_async(prompt, f"main_{sample_path}", max_tokens=500)
            api_calls.append(('main_prediction', main_call))
        
        # Execute the first batch of API calls
        results = {}
        if api_calls:
            responses = await asyncio.gather(*[call[1] for call in api_calls])
            for i, (call_type, _) in enumerate(api_calls):
                results[call_type] = responses[i]
        
        # Handle different reasoning modes
        if iterative_reasoning:
            # Parse category selection
            if 'category_selection' in results and results['category_selection']['success']:
                category_response = results['category_selection']['response']
                selected_categories = self.parse_selected_categories(category_response, num_categories)
                category_correct = self.evaluate_category_selection(selected_categories, disease_category)
                
                # Do iterative reasoning
                reasoning_result = await self.iterative_reasoning_with_flowcharts_async(
                    sample, num_inputs, selected_categories, max_reasoning_steps
                )
                
                predicted = reasoning_result['final_diagnosis']
                reasoning_trace = reasoning_result['reasoning_trace']
                reasoning_steps = reasoning_result['reasoning_steps']
                reasoning_path_correct = self.evaluate_reasoning_path(reasoning_trace, disease_category)
            else:
                # Fallback if category selection failed
                category_response = ""
                selected_categories = []
                category_correct = False
                predicted = ""
                reasoning_trace = []
                reasoning_steps = 0
                reasoning_path_correct = False
                
        elif two_step_reasoning:
            # Parse category selection
            if 'category_selection' in results and results['category_selection']['success']:
                category_response = results['category_selection']['response']
                selected_categories = self.parse_selected_categories(category_response, num_categories)
                category_correct = self.evaluate_category_selection(selected_categories, disease_category)
                
                # Step 2: Final diagnosis with flowcharts
                final_prompt = self.create_two_step_final_prompt(sample, num_inputs, selected_categories)
                # Final diagnosis reasoning needs generous token limit  
                predicted = self.query_llm(final_prompt, max_tokens=600)
            else:
                # Fallback if category selection failed
                category_response = ""
                selected_categories = []
                category_correct = False
                predicted = ""
                
        else:
            # Standard single-step evaluation
            if 'main_prediction' in results and results['main_prediction']['success']:
                predicted = results['main_prediction']['response']
            else:
                predicted = ""
            
            # No category selection in single-step mode
            selected_categories = []
            category_response = ""
            category_correct = None
        
        # Normalize and match prediction
        matched_prediction = self.find_best_match(predicted)
        
        # Determine correctness (use sync version for now, could optimize later)
        if self.use_llm_judge and predicted:
            # For now, use sync version - could make this async too
            correct = self.llm_judge_evaluation(matched_prediction, ground_truth)
        else:
            correct = ground_truth == matched_prediction
        
        # Build result
        result = {
            'sample_path': sample_path,
            'ground_truth': ground_truth,
            'disease_category': disease_category,
            'predicted_raw': predicted,
            'predicted_matched': matched_prediction,
            'correct': correct,
            'evaluation_method': 'llm_judge' if self.use_llm_judge else 'exact_match'
        }
        
        # Add mode-specific information
        if iterative_reasoning:
            result.update({
                'iterative_reasoning': True,
                'category_selection_response': category_response,
                'selected_categories': selected_categories,
                'category_selection_correct': category_correct,
                'reasoning_trace': reasoning_trace,
                'reasoning_steps': reasoning_steps,
                'reasoning_path_correct': reasoning_path_correct,
                'category_prompt': category_prompt,
                'initial_category_selection_response': reasoning_result.get('initial_category_selection_response', ''),
                'chosen_category': reasoning_result.get('category_used', '')
            })
        elif two_step_reasoning:
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

    async def evaluate_dataset_concurrent(self, num_inputs: int = 6,
                                         provide_diagnosis_list: bool = True, 
                                         max_samples: Optional[int] = None,
                                         samples_dir: str = None,
                                         two_step_reasoning: bool = False,
                                         num_categories: int = 3,
                                         iterative_reasoning: bool = False,
                                         max_reasoning_steps: int = 5) -> Dict:
        """
        Concurrent version of evaluate_dataset for faster processing
        """
        
        # Get samples directory
        if samples_dir is None:
            samples_dir = get_samples_directory(self.samples_dir)
        
        # Collect all sample files
        sample_files = collect_sample_files(samples_dir)
        
        if max_samples:
            sample_files = sample_files[:max_samples]
        
        print(f"🚀 Concurrent evaluation of {len(sample_files)} samples...")
        print(f"Using {num_inputs} input fields")
        print(f"Providing diagnosis list: {provide_diagnosis_list}")
        if two_step_reasoning:
            print(f"Two-step reasoning enabled (selecting {num_categories} categories)")
        elif iterative_reasoning:
            print(f"Iterative reasoning enabled (selecting {num_categories} categories, max {max_reasoning_steps} steps)")
        print(f"Max concurrent requests: {self.max_concurrent}")
        print(f"Rate limit: {self.rate_limiter.requests_per_minute} requests/minute")
        print(f"LLM Judge enabled: {self.use_llm_judge}")
        print()
        
        start_time = time.time()
        
        # Process samples in batches to avoid overwhelming the API
        batch_size = self.max_concurrent
        results = []
        
        for i in range(0, len(sample_files), batch_size):
            batch = sample_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(sample_files) + batch_size - 1) // batch_size
            
            print(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} samples)")
            
            # Create async tasks for this batch
            tasks = []
            for sample_path in batch:
                task = self.evaluate_sample_async(
                    sample_path, 
                    num_inputs, 
                    provide_diagnosis_list,
                    two_step_reasoning=two_step_reasoning,
                    num_categories=num_categories,
                    iterative_reasoning=iterative_reasoning,
                    max_reasoning_steps=max_reasoning_steps
                )
                tasks.append(task)
            
            # Execute batch concurrently
            batch_start = time.time()
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle any exceptions
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        print(f"❌ Error processing {batch[j]}: {result}")
                        # Create a failed result
                        result = {
                            'sample_path': batch[j],
                            'ground_truth': extract_diagnosis_from_path(batch[j]),
                            'disease_category': extract_disease_category_from_path(batch[j]),
                            'predicted_raw': "",
                            'predicted_matched': "",
                            'correct': False,
                            'evaluation_method': 'failed',
                            'error': str(result)
                        }
                    results.append(result)
                
                batch_time = time.time() - batch_start
                processed_so_far = min(len(sample_files), i + batch_size)
                elapsed = time.time() - start_time
                rate = processed_so_far / elapsed if elapsed > 0 else 0
                
                print(f"   ✅ Batch completed in {batch_time:.1f}s")
                print(f"   📊 Progress: {processed_so_far}/{len(sample_files)} ({processed_so_far/len(sample_files)*100:.1f}%)")
                print(f"   ⚡ Rate: {rate:.1f} samples/second")
                
                if processed_so_far < len(sample_files):
                    eta = (len(sample_files) - processed_so_far) / rate if rate > 0 else 0
                    print(f"   ⏱️  ETA: {eta/60:.1f} minutes")
                print()
                
            except Exception as e:
                print(f"❌ Batch {batch_num} failed: {e}")
                # Add failed results for this batch
                for sample_path in batch:
                    result = {
                        'sample_path': sample_path,
                        'ground_truth': extract_diagnosis_from_path(sample_path),
                        'disease_category': extract_disease_category_from_path(sample_path),
                        'predicted_raw': "",
                        'predicted_matched': "",
                        'correct': False,
                        'evaluation_method': 'failed',
                        'error': str(e)
                    }
                    results.append(result)
        
        total_time = time.time() - start_time
        print(f"🎯 Concurrent evaluation completed in {total_time/60:.1f} minutes!")
        print(f"📈 Average rate: {len(sample_files)/total_time:.1f} samples/second")
        print()
        
        # Calculate metrics (same as sync version)
        return self._calculate_metrics(results, num_inputs, provide_diagnosis_list, 
                                     two_step_reasoning, iterative_reasoning, 
                                     num_categories, max_reasoning_steps)
    
    def _calculate_metrics(self, results: List[Dict], num_inputs: int, 
                          provide_diagnosis_list: bool, two_step_reasoning: bool,
                          iterative_reasoning: bool, num_categories: int, 
                          max_reasoning_steps: int) -> Dict:
        """Helper method to calculate metrics from results"""
        
        # Filter out failed results for metrics calculation
        valid_results = [r for r in results if r.get('evaluation_method') != 'failed']
        
        if not valid_results:
            print("⚠️  Warning: No valid results for metrics calculation")
            return {
                'overall_metrics': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'num_samples': 0},
                'per_class_metrics': {},
                'category_metrics': {},
                'detailed_results': results,
                'configuration': {}
            }
        
        # Calculate overall metrics
        y_true = [r['ground_truth'] for r in valid_results]
        y_pred = [r['predicted_matched'] for r in valid_results]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # For multi-class precision/recall
        unique_labels = list(set(y_true + y_pred))
        
        precision = precision_score(y_true, y_pred, labels=unique_labels, 
                                  average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, labels=unique_labels, 
                            average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=unique_labels, 
                     average='weighted', zero_division=0)
        
        # Calculate mode-specific metrics
        overall_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': len(valid_results),
            'failed_samples': len(results) - len(valid_results)
        }
        
        if two_step_reasoning or iterative_reasoning:
            category_selections = [r.get('category_selection_correct') for r in valid_results 
                                 if r.get('category_selection_correct') is not None]
            if category_selections:
                category_selection_accuracy = sum(category_selections) / len(category_selections)
                overall_metrics['category_selection_accuracy'] = category_selection_accuracy
        
        if iterative_reasoning:
            # Calculate reasoning path metrics
            reasoning_paths = [r.get('reasoning_path_correct') for r in valid_results 
                             if r.get('reasoning_path_correct') is not None]
            if reasoning_paths:
                reasoning_path_accuracy = sum(reasoning_paths) / len(reasoning_paths)
                overall_metrics['reasoning_path_accuracy'] = reasoning_path_accuracy
            
            # Calculate average reasoning steps
            reasoning_steps = [r.get('reasoning_steps', 0) for r in valid_results 
                             if r.get('reasoning_steps') is not None]
            if reasoning_steps:
                avg_reasoning_steps = sum(reasoning_steps) / len(reasoning_steps)
                overall_metrics['avg_reasoning_steps'] = avg_reasoning_steps
        
        # Per-class metrics
        per_class_metrics = {}
        for label in set(y_true):
            label_true = [1 if gt == label else 0 for gt in y_true]
            label_pred = [1 if pred == label else 0 for pred in y_pred]
            
            if sum(label_true) > 0:
                per_class_metrics[label] = {
                    'precision': precision_score(label_true, label_pred, zero_division=0),
                    'recall': recall_score(label_true, label_pred, zero_division=0),
                    'f1': f1_score(label_true, label_pred, zero_division=0),
                    'support': sum(label_true)
                }
        
        # Calculate disease category metrics
        category_metrics = self.calculate_category_metrics(valid_results)
        
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
                'iterative_reasoning': iterative_reasoning,
                'num_categories': num_categories if (two_step_reasoning or iterative_reasoning) else None,
                'max_reasoning_steps': max_reasoning_steps if iterative_reasoning else None,
                'max_concurrent': self.max_concurrent
            }
        } 