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
                 thinking_mode: bool = True, llm_test_overlap: bool = False):
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
            llm_test_overlap: Whether to use LLM for test overlap calculation
        """
        self.flowchart_dir = flowchart_dir
        self.samples_dir = samples_dir
        self.possible_diagnoses = load_possible_diagnoses(flowchart_dir)
        self.flowchart_categories = load_flowchart_categories(flowchart_dir)
        self.use_llm_judge = use_llm_judge
        self.show_responses = show_responses
        self.max_concurrent = max_concurrent
        self.llm_test_overlap = llm_test_overlap
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
                       max_reasoning_steps: int = 5, progressive_reasoning: bool = False,
                       num_suspicions: int = 3, progressive_fast_mode: bool = False) -> Dict:
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
            progressive_reasoning: Whether to use progressive reasoning
            num_suspicions: Number of suspicions to generate in the first stage
            progressive_fast_mode: If True, use fast mode for progressive reasoning
        
        Returns:
            Dict with evaluation results
        """
        sample = load_sample(sample_path)
        ground_truth = extract_diagnosis_from_path(sample_path)
        disease_category = extract_disease_category_from_path(sample_path)
        
        if progressive_reasoning:
            # Progressive reasoning: History -> Suspicions -> Tests -> Final diagnosis
            progressive_result = self.progressive_reasoning_workflow(
                sample, num_suspicions, max_reasoning_steps, fast_mode=progressive_fast_mode
            )
            
            predicted = progressive_result['final_diagnosis']
            reasoning_trace = progressive_result['reasoning_trace']
            suspicions_generated = progressive_result['suspicions']
            recommended_tests = progressive_result['recommended_tests']
            chosen_suspicion = progressive_result['chosen_suspicion']
            reasoning_steps = progressive_result['reasoning_steps']
            
            # CRITICAL: Calculate test overlap metrics for progressive reasoning
            test_overlap_metrics = self.calculate_test_overlap_metrics(recommended_tests, sample)
            
            if self.show_responses:
                print(f"Sample: {sample_path}")
                print(f"Disease Category: {disease_category}")
                print(f"Progressive Reasoning Workflow:")
                print(f"  Stage 1 - Initial Suspicions: {suspicions_generated}")
                print(f"  Stage 2 - Recommended Tests: {recommended_tests[:100]}...")
                print(f"  Stage 3 - Chosen Suspicion: {chosen_suspicion}")
                print(f"  Stage 4 - Reasoning Steps: {reasoning_steps}")
                print(f"Test Overlap Metrics:")
                print(f"  Precision: {test_overlap_metrics['test_overlap_precision']:.3f} (avoiding unnecessary tests)")
                print(f"  Recall: {test_overlap_metrics['test_overlap_recall']:.3f} (not missing necessary tests)")
                print(f"  F1-Score: {test_overlap_metrics['test_overlap_f1']:.3f}")
                print(f"  Recommended: {test_overlap_metrics['tests_recommended_count']}, Actual: {test_overlap_metrics['tests_actual_count']}, Overlap: {test_overlap_metrics['tests_overlap_count']}")
                if test_overlap_metrics['unnecessary_tests_list']:
                    print(f"  Unnecessary: {test_overlap_metrics['unnecessary_tests_list']}")
                if test_overlap_metrics['missed_tests_list']:
                    print(f"  Missed: {test_overlap_metrics['missed_tests_list']}")
                print(f"Final Diagnosis: '{predicted}'")
                print(f"Ground Truth: '{ground_truth}'")
                print()
        
        elif iterative_reasoning:
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
                print(f"Ground Truth: '{ground_truth}'")
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
        if progressive_reasoning:
            result.update({
                'progressive_reasoning': True,
                'suspicions_generated': suspicions_generated,
                'recommended_tests': recommended_tests,
                'chosen_suspicion': chosen_suspicion,
                'reasoning_trace': reasoning_trace,
                'reasoning_steps': reasoning_steps,
                'test_overlap_metrics': test_overlap_metrics,
                'prompts_and_responses': progressive_result.get('prompts_and_responses', []),
                'progressive_mode': progressive_result.get('mode', 'unknown'),
                'prompt': f"Progressive Reasoning ({progressive_result.get('mode', 'unknown')} mode) - {len(progressive_result.get('prompts_and_responses', []))} stages"
            })
        elif iterative_reasoning:
            result.update({
                'iterative_reasoning': True,
                'category_selection_response': category_response,
                'selected_categories': selected_categories,
                'category_selection_correct': category_correct,
                'reasoning_trace': reasoning_trace,
                'reasoning_steps': reasoning_steps,
                'reasoning_path_correct': reasoning_path_correct,
                'category_prompt': category_prompt
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
                        max_reasoning_steps: int = 5,
                        progressive_reasoning: bool = False,
                        num_suspicions: int = 3,
                        progressive_fast_mode: bool = False) -> Dict:
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
            progressive_reasoning: Whether to use progressive reasoning
            num_suspicions: Number of suspicions to generate in the first stage
            progressive_fast_mode: If True, use fast mode for progressive reasoning
        
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
        if progressive_reasoning:
            print(f"Progressive reasoning enabled (generating {num_suspicions} suspicions, max {max_reasoning_steps} steps)")
            print(f"Workflow: History -> Suspicions -> Tests -> Choice -> Flowcharts")
        elif two_step_reasoning:
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
                max_reasoning_steps=max_reasoning_steps,
                progressive_reasoning=progressive_reasoning,
                num_suspicions=num_suspicions,
                progressive_fast_mode=progressive_fast_mode
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
        
        if progressive_reasoning:
            # Calculate test overlap metrics
            test_overlap_precision_scores = [r.get('test_overlap_metrics', {}).get('test_overlap_precision', 0) for r in results 
                                           if r.get('test_overlap_metrics') is not None]
            test_overlap_recall_scores = [r.get('test_overlap_metrics', {}).get('test_overlap_recall', 0) for r in results 
                                        if r.get('test_overlap_metrics') is not None]
            test_overlap_f1_scores = [r.get('test_overlap_metrics', {}).get('test_overlap_f1', 0) for r in results 
                                    if r.get('test_overlap_metrics') is not None]
            test_overlap_jaccard_scores = [r.get('test_overlap_metrics', {}).get('test_overlap_jaccard', 0) for r in results 
                                         if r.get('test_overlap_metrics') is not None]
            
            # Calculate overall test overlap performance
            if test_overlap_precision_scores:
                overall_metrics['test_overlap_precision'] = sum(test_overlap_precision_scores) / len(test_overlap_precision_scores)
                overall_metrics['test_overlap_recall'] = sum(test_overlap_recall_scores) / len(test_overlap_recall_scores)
                overall_metrics['test_overlap_f1'] = sum(test_overlap_f1_scores) / len(test_overlap_f1_scores)
                overall_metrics['test_overlap_jaccard'] = sum(test_overlap_jaccard_scores) / len(test_overlap_jaccard_scores)
            
            # Calculate average test counts
            recommended_counts = [r.get('test_overlap_metrics', {}).get('tests_recommended_count', 0) for r in results 
                                if r.get('test_overlap_metrics') is not None]
            actual_counts = [r.get('test_overlap_metrics', {}).get('tests_actual_count', 0) for r in results 
                           if r.get('test_overlap_metrics') is not None]
            overlap_counts = [r.get('test_overlap_metrics', {}).get('tests_overlap_count', 0) for r in results 
                            if r.get('test_overlap_metrics') is not None]
            unnecessary_counts = [r.get('test_overlap_metrics', {}).get('unnecessary_tests_count', 0) for r in results 
                                if r.get('test_overlap_metrics') is not None]
            missed_counts = [r.get('test_overlap_metrics', {}).get('missed_tests_count', 0) for r in results 
                           if r.get('test_overlap_metrics') is not None]
            
            if recommended_counts:
                overall_metrics['avg_tests_recommended'] = sum(recommended_counts) / len(recommended_counts)
                overall_metrics['avg_tests_actual'] = sum(actual_counts) / len(actual_counts)
                overall_metrics['avg_tests_overlap'] = sum(overlap_counts) / len(overlap_counts)
                overall_metrics['avg_unnecessary_tests'] = sum(unnecessary_counts) / len(unnecessary_counts)
                overall_metrics['avg_missed_tests'] = sum(missed_counts) / len(missed_counts)
            
            # Calculate average reasoning steps for progressive reasoning
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
                'progressive_reasoning': progressive_reasoning,
                'two_step_reasoning': two_step_reasoning,
                'iterative_reasoning': iterative_reasoning,
                'num_categories': num_categories if (two_step_reasoning or iterative_reasoning) else None,
                'num_suspicions': num_suspicions if progressive_reasoning else None,
                'max_reasoning_steps': max_reasoning_steps if (iterative_reasoning or progressive_reasoning) else None
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
        
        # Start iterative reasoning
        current_step = 1
        
        # Step 1: Select starting category and root node
        if len(selected_categories) == 1:
            current_category = selected_categories[0]
        else:
            # Let LLM choose which category to start with
            category_prompt = self.create_initial_category_selection_prompt(
                patient_summary, selected_categories, flowcharts
            )
            category_response = self.query_llm(category_prompt, max_tokens=200)
            current_category = self.parse_category_selection(category_response, selected_categories)
        
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
            'step': current_step,
            'category': current_category,
            'current_node': current_node,
            'action': 'start',
            'response': f"Starting with {current_category} -> {current_node}"
        })
        
        # Iterative reasoning through the flowchart
        while current_step < max_steps:
            current_step += 1
            
            # Check if current node is a leaf (final diagnosis)
            if is_leaf_diagnosis(flowcharts[current_category]['structure'], current_node):
                final_diagnosis = current_node
                reasoning_trace.append({
                    'step': current_step,
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
                    'step': current_step,
                    'category': current_category,
                    'current_node': current_node,
                    'action': 'final_diagnosis',
                    'response': f"No further options, final diagnosis: {current_node}"
                })
                break
            
            # Create reasoning step prompt
            step_prompt = format_reasoning_step(
                current_step, 
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
                'step': current_step,
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
            'category_used': current_category
        }
    
    def create_initial_category_selection_prompt(self, patient_summary: str, 
                                               categories: List[str], 
                                               flowcharts: Dict) -> str:
        """Create prompt for selecting which category to start reasoning with"""
        
        prompt = "You are a medical expert beginning diagnostic reasoning. "
        prompt += "Based on the patient information, select which disease category to explore first.\n\n"
        
        prompt += f"Patient Information:\n{patient_summary}\n\n"
        
        prompt += "Available categories to explore:\n"
        for i, category in enumerate(categories, 1):
            prompt += f"{i}. {category}\n"
        
        prompt += f"\nSelect the most promising category to start your diagnostic reasoning. "
        prompt += f"Respond with ONLY the number (1-{len(categories)}) of your choice.\n"
        
        return prompt
    
    def parse_category_selection(self, response: str, categories: List[str]) -> str:
        """Parse category selection response"""
        
        import re
        number_match = re.search(r'\b(\d+)\b', response.strip())
        if number_match:
            try:
                choice_num = int(number_match.group(1))
                if 1 <= choice_num <= len(categories):
                    return categories[choice_num - 1]
            except ValueError:
                pass
        
        # Fallback to first category
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
        
        # Start iterative reasoning
        current_step = 1
        
        # Step 1: Select starting category (may need async call)
        if len(selected_categories) == 1:
            current_category = selected_categories[0]
        else:
            # Async category selection
            category_prompt = self.create_initial_category_selection_prompt(
                patient_summary, selected_categories, flowcharts
            )
            result = await self.query_llm_async(category_prompt, f"category_selection", max_tokens=200)
            if result['success']:
                current_category = self.parse_category_selection(result['response'], selected_categories)
            else:
                current_category = selected_categories[0]  # Fallback
        
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
        
        # CRITICAL FIX: Replace hardcoded "starting" message with actual LLM reasoning
        # The LLM should analyze clinical evidence against flowchart knowledge to justify category selection
        initial_reasoning_prompt = self.create_initial_reasoning_prompt(
            patient_summary, current_category, flowcharts[current_category]['knowledge'], current_node
        )
        initial_reasoning_result = await self.query_llm_async(initial_reasoning_prompt, f"initial_reasoning", max_tokens=800)
        
        reasoning_trace.append({
            'step': current_step,
            'category': current_category,
            'current_node': current_node,
            'action': 'initial_reasoning',
            'prompt': initial_reasoning_prompt,
            'response': initial_reasoning_result['response'] if initial_reasoning_result['success'] else "",
            'reasoning_type': 'clinical_evidence_analysis'
        })
        
        # Iterative reasoning through the flowchart
        while current_step < max_steps:
            current_step += 1
            
            # Check if current node is a leaf (final diagnosis)
            if is_leaf_diagnosis(flowcharts[current_category]['structure'], current_node):
                final_diagnosis = current_node
                reasoning_trace.append({
                    'step': current_step,
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
                    'step': current_step,
                    'category': current_category,
                    'current_node': current_node,
                    'action': 'final_diagnosis',
                    'response': f"No further options, final diagnosis: {current_node}"
                })
                break
            
            # Create reasoning step prompt
            step_prompt = format_reasoning_step(
                current_step, 
                current_node, 
                children,
                flowcharts[current_category]['knowledge'],
                patient_summary
            )
            
            # Async API call for reasoning step
            # Iterative reasoning steps need large token limit for evidence matching and comparative analysis
            result = await self.query_llm_async(step_prompt, f"reasoning_step_{current_step}", max_tokens=1200)
            
            if result['success']:
                step_response = result['response']
                
                # Parse the choice and reasoning
                reasoning_result = extract_reasoning_choice(step_response, children)
                chosen_node = reasoning_result['chosen_option']
                
                reasoning_trace.append({
                    'step': current_step,
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
            else:
                # API call failed, break the reasoning
                print(f"API call failed at step {current_step}: {result['error']}")
                break
        
        # If we didn't reach a final diagnosis, use the last node
        if not final_diagnosis:
            final_diagnosis = current_node
        
        return {
            'final_diagnosis': final_diagnosis,
            'reasoning_trace': reasoning_trace,
            'reasoning_steps': len(reasoning_trace),
            'reasoning_successful': final_diagnosis != "",
            'category_used': current_category
        }
    
    async def evaluate_sample_async(self, sample_path: str, num_inputs: int, 
                                   provide_diagnosis_list: bool, two_step_reasoning: bool = False,
                                   num_categories: int = 3, iterative_reasoning: bool = False,
                                   max_reasoning_steps: int = 5, progressive_reasoning: bool = False,
                                   num_suspicions: int = 3, progressive_fast_mode: bool = False) -> Dict:
        """
        Async version of evaluate_sample for concurrent processing
        """
        sample = load_sample(sample_path)
        ground_truth = extract_diagnosis_from_path(sample_path)
        disease_category = extract_disease_category_from_path(sample_path)
        
        # Prepare API calls that need to be made
        api_calls = []
        
        if progressive_reasoning:
            # Progressive reasoning workflow - multiple sequential stages
            # For now, use synchronous approach within async function
            # Could be optimized further for concurrent API calls
            progressive_result = self.progressive_reasoning_workflow(
                sample, num_suspicions, max_reasoning_steps, fast_mode=progressive_fast_mode
            )
            
            predicted = progressive_result['final_diagnosis']
            suspicions_generated = progressive_result['suspicions']
            recommended_tests = progressive_result['recommended_tests']
            chosen_suspicion = progressive_result['chosen_suspicion']
            reasoning_trace = progressive_result['reasoning_trace']
            reasoning_steps = progressive_result['reasoning_steps']
            
            # Calculate test overlap metrics for progressive reasoning
            test_overlap_metrics = self.calculate_test_overlap_metrics(recommended_tests, sample)
            
        elif iterative_reasoning:
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
        if progressive_reasoning:
            # Results already available from progressive workflow
            # No additional API calls needed - results are in variables above
            selected_categories = []
            category_response = ""
            category_correct = None
            
        elif iterative_reasoning:
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
        if progressive_reasoning:
            result.update({
                'progressive_reasoning': True,
                'suspicions_generated': suspicions_generated,
                'recommended_tests': recommended_tests,
                'chosen_suspicion': chosen_suspicion,
                'reasoning_trace': reasoning_trace,
                'reasoning_steps': reasoning_steps,
                'test_overlap_metrics': test_overlap_metrics,
                'prompts_and_responses': progressive_result.get('prompts_and_responses', []),
                'progressive_mode': progressive_result.get('mode', 'unknown'),
                'prompt': f"Progressive Reasoning ({progressive_result.get('mode', 'unknown')} mode) - {len(progressive_result.get('prompts_and_responses', []))} stages"
            })
        elif iterative_reasoning:
            result.update({
                'iterative_reasoning': True,
                'category_selection_response': category_response,
                'selected_categories': selected_categories,
                'category_selection_correct': category_correct,
                'reasoning_trace': reasoning_trace,
                'reasoning_steps': reasoning_steps,
                'reasoning_path_correct': reasoning_path_correct,
                'category_prompt': category_prompt if 'category_prompt' in locals() else ""
            })
        elif two_step_reasoning:
            result.update({
                'two_step_reasoning': True,
                'category_selection_response': category_response,
                'selected_categories': selected_categories,
                'category_selection_correct': category_correct,
                'category_prompt': category_prompt if 'category_prompt' in locals() else "",
                'final_prompt': final_prompt if 'final_prompt' in locals() else ""
            })
        else:
            result['prompt'] = prompt if 'prompt' in locals() else ""
        
        return result 

    async def evaluate_dataset_concurrent(self, num_inputs: int = 6,
                                         provide_diagnosis_list: bool = True, 
                                         max_samples: Optional[int] = None,
                                         samples_dir: str = None,
                                         two_step_reasoning: bool = False,
                                         num_categories: int = 3,
                                         iterative_reasoning: bool = False,
                                         max_reasoning_steps: int = 5,
                                         progressive_reasoning: bool = False,
                                         num_suspicions: int = 3,
                                         progressive_fast_mode: bool = False) -> Dict:
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
                    max_reasoning_steps=max_reasoning_steps,
                    progressive_reasoning=progressive_reasoning,
                    num_suspicions=num_suspicions,
                    progressive_fast_mode=progressive_fast_mode
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
                                     num_categories, max_reasoning_steps,
                                     progressive_reasoning, num_suspicions)
    
    def _calculate_metrics(self, results: List[Dict], num_inputs: int, 
                          provide_diagnosis_list: bool, two_step_reasoning: bool,
                          iterative_reasoning: bool, num_categories: int, 
                          max_reasoning_steps: int, progressive_reasoning: bool = False,
                          num_suspicions: int = 3) -> Dict:
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
                'progressive_reasoning': progressive_reasoning,
                'two_step_reasoning': two_step_reasoning,
                'iterative_reasoning': iterative_reasoning,
                'num_categories': num_categories if (two_step_reasoning or iterative_reasoning) else None,
                'num_suspicions': num_suspicions if progressive_reasoning else None,
                'max_reasoning_steps': max_reasoning_steps if (iterative_reasoning or progressive_reasoning) else None
            }
        } 

    def progressive_reasoning_workflow(self, sample: Dict, num_suspicions: int = 3, 
                                     max_reasoning_steps: int = 5, fast_mode: bool = False) -> Dict:
        """
        Progressive clinical workflow reasoning:
        Stage 1: History (inputs 1-4) -> Generate top suspicions + recommended tests
        Stage 2: Add test results (inputs 5-6) -> Choose suspicion  
        Stage 3: Follow flowcharts iteratively to final diagnosis
        
        Args:
            sample: The clinical sample data
            num_suspicions: Number of initial suspicions to generate
            max_reasoning_steps: Maximum reasoning steps in final stage
            fast_mode: If True, combine stages for faster processing
        
        Returns:
            Dict with progressive reasoning results
        """
        
        if fast_mode:
            # Fast mode: Combine stages 1-3 into a single API call
            return self._progressive_reasoning_fast(sample, num_suspicions, max_reasoning_steps)
        else:
            # Standard mode: Full 4-stage workflow
            return self._progressive_reasoning_standard(sample, num_suspicions, max_reasoning_steps)
    
    def _progressive_reasoning_fast(self, sample: Dict, num_suspicions: int, max_reasoning_steps: int) -> Dict:
        """Fast progressive reasoning - combines multiple stages"""
        
        # Get history and full clinical information
        history_summary = self.create_history_summary(sample)
        full_summary = self.create_patient_data_summary(sample, 6)
        
        # Single combined prompt for stages 1-3
        combined_prompt = f"""You are a medical expert following a progressive clinical workflow.

**STAGE 1 - Initial Assessment (History Only):**
{history_summary}

**Available Disease Categories:**
{chr(10).join(f"{i}. {cat}" for i, cat in enumerate(self.flowchart_categories, 1))}

Based on this history, select {num_suspicions} most likely disease categories from the list above.

**STAGE 2 - Test Planning:**
For your chosen categories, what physical exam and lab/imaging tests would be most helpful to differentiate between them?

**STAGE 3 - Final Assessment (Complete Information):**
{full_summary}

Now with complete clinical information available, choose your most likely disease category and then provide a specific diagnosis.

**Possible Primary Discharge Diagnoses:**"""
        for i, diagnosis in enumerate(self.possible_diagnoses, 1):
            combined_prompt += f"\n{i}. {diagnosis}"
        
        combined_prompt += f"""

**INSTRUCTIONS:**
• Choose categories from the disease categories list above
• IMPORTANT: Your final diagnosis MUST be selected from the possible diagnoses list above
• Use the complete clinical information to make your final diagnosis

**FORMAT:**
**INITIAL CATEGORY SUSPICIONS:** [List {num_suspicions} categories]
**RECOMMENDED TESTS:** [Brief list of key tests]
**CHOSEN CATEGORY:** [Best category from suspicions]
**FINAL DIAGNOSIS:** [Exact diagnosis name from the possible diagnoses list above]
**REASONING:** [Brief explanation for final diagnosis]"""
        
        # Single API call
        response = self.query_llm(combined_prompt, max_tokens=800)
        
        # Parse response
        import re
        
        # Extract category suspicions
        suspicions_match = re.search(r'INITIAL CATEGORY SUSPICIONS:\s*(.*?)(?=\*\*RECOMMENDED TESTS:|$)', response, re.DOTALL | re.IGNORECASE)
        suspicions_text = suspicions_match.group(1).strip() if suspicions_match else ""
        suspicions = self._parse_suspicions_from_text(suspicions_text, num_suspicions)
        
        # Extract recommended tests
        tests_match = re.search(r'RECOMMENDED TESTS:\s*(.*?)(?=\*\*CHOSEN CATEGORY:|$)', response, re.DOTALL | re.IGNORECASE)
        recommended_tests = tests_match.group(1).strip() if tests_match else ""
        
        # Extract chosen category
        category_match = re.search(r'CHOSEN CATEGORY:\s*(.*?)(?=\*\*FINAL DIAGNOSIS:|$)', response, re.DOTALL | re.IGNORECASE)
        chosen_category = category_match.group(1).strip() if category_match else ""
        
        # Extract final diagnosis
        diagnosis_match = re.search(r'FINAL DIAGNOSIS:\s*(.*?)(?=\*\*REASONING:|$)', response, re.DOTALL | re.IGNORECASE)
        final_diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else ""
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.*?)$', response, re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Match final diagnosis against possible diagnoses
        matched_diagnosis = self.find_best_match(final_diagnosis)
        
        # Create reasoning trace
        reasoning_trace = [
            {
                'step': 1,
                'action': 'combined_progressive',
                'prompt': combined_prompt,
                'response': response,
                'suspicions': suspicions,
                'recommended_tests': recommended_tests,
                'chosen_category': chosen_category,
                'final_diagnosis': final_diagnosis,
                'matched_diagnosis': matched_diagnosis,
                'reasoning': reasoning
            }
        ]
        
        # CRITICAL: Collect all prompts and responses for analysis
        prompts_and_responses = [
            {
                'stage': 'combined_fast_mode',
                'prompt': combined_prompt,
                'response': response,
                'parsed_suspicions': suspicions,
                'parsed_tests': recommended_tests,
                'parsed_choice': chosen_category,
                'parsed_diagnosis': final_diagnosis,
                'parsed_reasoning': reasoning
            }
        ]
        
        return {
            'final_diagnosis': matched_diagnosis,
            'reasoning_trace': reasoning_trace,
            'reasoning_steps': 1,
            'suspicions': suspicions,
            'recommended_tests': recommended_tests,
            'chosen_suspicion': chosen_category,  # The chosen category
            'reasoning_successful': bool(matched_diagnosis),
            'prompts_and_responses': prompts_and_responses,
            'mode': 'fast'
        }
    
    def _progressive_reasoning_standard(self, sample: Dict, num_suspicions: int, max_reasoning_steps: int) -> Dict:
        """Standard progressive reasoning - full 4-stage workflow"""
        
        # CRITICAL: Track all prompts and responses for analysis
        prompts_and_responses = []
        
        # Stage 1: Generate suspicions based on history only (inputs 1-4)
        history_summary = self.create_history_summary(sample)
        suspicions_prompt = self.create_suspicions_prompt(history_summary, num_suspicions)
        suspicions_response = self.query_llm(suspicions_prompt, max_tokens=600)
        suspicions = self.parse_suspicions(suspicions_response, num_suspicions)
        
        prompts_and_responses.append({
            'stage': 'stage_1_suspicions',
            'prompt': suspicions_prompt,
            'response': suspicions_response,
            'parsed_suspicions': suspicions,
            'history_summary': history_summary
        })
        
        # Stage 2: Generate recommended tests based on suspicions
        tests_prompt = self.create_tests_recommendation_prompt(history_summary, suspicions)
        tests_response = self.query_llm(tests_prompt, max_tokens=400)
        recommended_tests = tests_response.strip()
        
        prompts_and_responses.append({
            'stage': 'stage_2_tests',
            'prompt': tests_prompt,
            'response': tests_response,
            'parsed_tests': recommended_tests
        })
        
        # Stage 3: Present test results and choose suspicion
        full_summary = self.create_patient_data_summary(sample, 6)  # All 6 inputs
        suspicion_choice_prompt = self.create_suspicion_choice_prompt(
            history_summary, full_summary, suspicions, recommended_tests
        )
        choice_response = self.query_llm(suspicion_choice_prompt, max_tokens=600)
        chosen_suspicion, choice_reasoning = self.parse_suspicion_choice(choice_response, suspicions)
        
        prompts_and_responses.append({
            'stage': 'stage_3_choice',
            'prompt': suspicion_choice_prompt,
            'response': choice_response,
            'parsed_choice': chosen_suspicion,
            'choice_reasoning': choice_reasoning,  # CRITICAL: Save the reasoning for why this choice was made
            'full_summary': full_summary
        })
        
        # Stage 4: Progressive iterative reasoning based on chosen suspicion
        reasoning_result = self.progressive_iterative_reasoning(
            sample, chosen_suspicion, suspicions, max_reasoning_steps
        )
        
        # Add the final reasoning prompts and responses if available
        if 'reasoning_trace' in reasoning_result:
            for step in reasoning_result['reasoning_trace']:
                if 'prompt' in step:
                    prompts_and_responses.append({
                        'stage': f'stage_4_reasoning_step_{step.get("step", "unknown")}',
                        'prompt': step['prompt'],
                        'response': step.get('response', ''),
                        'step_info': step
                    })
        
        # Include the main prompt used in final reasoning (if available)
        final_prompt = self.create_patient_data_summary(sample, 6)  # This would be used in iterative reasoning
        
        return {
            'final_diagnosis': reasoning_result['final_diagnosis'],
            'reasoning_trace': reasoning_result['reasoning_trace'],
            'reasoning_steps': reasoning_result['reasoning_steps'],
            'suspicions': suspicions,
            'recommended_tests': recommended_tests,
            'chosen_suspicion': chosen_suspicion,
            'reasoning_successful': reasoning_result['reasoning_successful'],
            'prompts_and_responses': prompts_and_responses,
            'mode': 'standard'
        }
    
    def _parse_suspicions_from_text(self, text: str, num_suspicions: int) -> List[str]:
        """Parse suspicions from combined response text"""
        
        suspicions = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        import re
        for line in lines:
            # Look for numbered lines, bullet points, or just diagnoses
            match = re.match(r'^(?:\d+\.|[•-])\s*(.+)', line)
            if match:
                diagnosis = match.group(1).strip()
                
                # Clean up markdown formatting
                diagnosis = re.sub(r'^\*+', '', diagnosis)  # Remove leading asterisks
                diagnosis = re.sub(r'\*+$', '', diagnosis)  # Remove trailing asterisks
                diagnosis = re.sub(r'^\*+(.+?)\*+$', r'\1', diagnosis)  # Remove surrounding asterisks
                diagnosis = diagnosis.strip()
                
                # Skip empty or invalid entries
                if diagnosis and not diagnosis.startswith('*') and len(diagnosis) > 1:
                    suspicions.append(diagnosis)
                    if len(suspicions) >= num_suspicions:
                        break
            elif line and not re.match(r'^[A-Z\s]+:', line):  # Not a section header
                clean_line = line.strip()
                # Clean up markdown formatting
                clean_line = re.sub(r'^\*+', '', clean_line)
                clean_line = re.sub(r'\*+$', '', clean_line)
                clean_line = clean_line.strip()
                
                if clean_line and not clean_line.startswith('*') and len(clean_line) > 1:
                    suspicions.append(clean_line)
                    if len(suspicions) >= num_suspicions:
                        break
        
        # Fill in with generic suspicions if we don't have enough
        while len(suspicions) < num_suspicions:
            suspicions.append(f"Suspicion {len(suspicions) + 1}")
        
        return suspicions[:num_suspicions]
    
    def create_history_summary(self, sample: Dict) -> str:
        """Create summary from first 4 inputs only (history)"""
        
        input_descriptions = {
            1: "Chief Complaint",
            2: "History of Present Illness", 
            3: "Past Medical History",
            4: "Family History"
        }
        
        summary = ""
        for i in range(1, 5):  # Only inputs 1-4
            input_key = f"input{i}"
            if input_key in sample:
                content = sample[input_key]
                summary += f"• {input_descriptions[i]}: {content}\n"
        
        return summary.strip()
    
    def create_suspicions_prompt(self, history_summary: str, num_suspicions: int) -> str:
        """Create prompt for generating initial suspicions based on history"""
        
        prompt = "You are a medical expert generating initial diagnostic category suspicions based on patient history.\n\n"
        
        prompt += f"**Patient History:**\n{history_summary}\n\n"
        
        # CRITICAL: Provide the list of disease categories (not specific diagnoses)
        prompt += f"**Available Disease Categories to Consider:**\n"
        for i, category in enumerate(self.flowchart_categories, 1):
            prompt += f"{i}. {category}\n"
        prompt += "\n"
        
        prompt += f"Based on the patient history above, select the {num_suspicions} most likely disease categories from the available categories list.\n\n"
        
        prompt += f"**Instructions:**\n"
        prompt += f"• Choose ONLY from the available disease categories listed above\n"
        prompt += f"• Consider the historical information provided\n"
        prompt += f"• Focus on the most likely disease categories given the presentation\n"
        prompt += f"• List category suspicions in order of likelihood\n"
        prompt += f"• Use the exact category names from the list\n\n"
        
        prompt += f"**Format:**\n"
        for i in range(1, num_suspicions + 1):
            prompt += f"{i}. [Exact category name from available list]\n"
        
        prompt += f"\nTop {num_suspicions} Disease Category Suspicions:"
        
        return prompt
    
    def create_tests_recommendation_prompt(self, history_summary: str, suspicions: List[str]) -> str:
        """Create prompt for recommending necessary tests"""
        
        prompt = "You are a medical expert determining the minimum necessary tests to differentiate between diagnostic suspicions.\n\n"
        
        prompt += f"**Patient History:**\n{history_summary}\n\n"
        
        prompt += f"**Current Diagnostic Suspicions:**\n"
        for i, suspicion in enumerate(suspicions, 1):
            prompt += f"{i}. {suspicion}\n"
        
        prompt += f"\n**Task:**\nDetermine the minimum necessary physical examination findings and laboratory/imaging tests needed to differentiate between these suspicions and establish a diagnosis.\n\n"
        
        prompt += f"**Instructions:**\n"
        prompt += f"• Focus on tests that would help distinguish between the listed suspicions\n"
        prompt += f"• Prioritize the most informative and cost-effective tests\n"
        prompt += f"• Include both physical exam components and lab/imaging studies\n"
        prompt += f"• Be specific and practical\n\n"
        
        prompt += f"**Recommended Tests and Examinations:**"
        
        return prompt
    
    def create_suspicion_choice_prompt(self, history_summary: str, full_summary: str, 
                                     suspicions: List[str], recommended_tests: str) -> str:
        """Create prompt for choosing suspicion after seeing test results"""
        
        prompt = "You are a medical expert reviewing test results to narrow down your diagnostic suspicions.\n\n"
        
        prompt += f"**Initial Assessment Based on History:**\n{history_summary}\n\n"
        
        prompt += f"**Initial Suspicions:**\n"
        for i, suspicion in enumerate(suspicions, 1):
            prompt += f"{i}. {suspicion}\n"
        
        prompt += f"\n**Now Available - Complete Clinical Information:**\n{full_summary}\n\n"
        
        prompt += f"**Task:**\nBased on the physical examination findings and test results now available, choose the most likely diagnosis from your initial suspicions.\n\n"
        
        prompt += f"**Instructions:**\n"
        prompt += f"• Compare the actual findings with what you would expect for each suspicion\n"
        prompt += f"• Choose the suspicion most consistent with the complete clinical picture\n"
        prompt += f"• Provide brief reasoning for your choice\n\n"
        
        prompt += f"**Format:**\n"
        prompt += f"**CHOSEN SUSPICION:** [Number] - [Diagnosis name]\n"
        prompt += f"**REASONING:** [Brief explanation based on findings]\n"
        
        return prompt
    
    def parse_suspicions(self, response: str, num_suspicions: int) -> List[str]:
        """Parse LLM response to extract suspicions list"""
        
        suspicions = []
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        import re
        for line in lines:
            # Look for numbered lines like "1. Diagnosis name"
            match = re.match(r'^\d+\.\s*(.+)', line)
            if match:
                diagnosis = match.group(1).strip()
                
                # Clean up markdown formatting
                diagnosis = re.sub(r'^\*+', '', diagnosis)  # Remove leading asterisks
                diagnosis = re.sub(r'\*+$', '', diagnosis)  # Remove trailing asterisks
                diagnosis = re.sub(r'^\*+(.+?)\*+$', r'\1', diagnosis)  # Remove surrounding asterisks
                diagnosis = diagnosis.strip()
                
                # Skip empty or invalid entries
                if diagnosis and not diagnosis.startswith('*') and len(diagnosis) > 1:
                    suspicions.append(diagnosis)
                    if len(suspicions) >= num_suspicions:
                        break
        
        # If we didn't find enough from numbered format, try other formats
        if len(suspicions) < num_suspicions:
            for line in lines:
                # Look for lines that might be diagnoses (not section headers)
                if not re.match(r'^[A-Z\s]+:', line) and not line.startswith('**') and not line.startswith('#'):
                    # Clean line
                    clean_line = re.sub(r'[•\-\*]\s*', '', line).strip()
                    clean_line = re.sub(r'^\*+', '', clean_line)
                    clean_line = re.sub(r'\*+$', '', clean_line)
                    clean_line = clean_line.strip()
                    
                    if clean_line and len(clean_line) > 2 and clean_line not in suspicions:
                        suspicions.append(clean_line)
                        if len(suspicions) >= num_suspicions:
                            break
        
        # Fill in with generic suspicions if we don't have enough
        while len(suspicions) < num_suspicions:
            suspicions.append(f"Suspicion {len(suspicions) + 1}")
        
        return suspicions[:num_suspicions]
    
    def parse_suspicion_choice(self, response: str, suspicions: List[str]) -> tuple:
        """Parse chosen suspicion and reasoning from response"""
        
        import re
        
        chosen_suspicion = None
        reasoning = ""
        
        # Try to extract from CHOSEN SUSPICION and REASONING sections
        chosen_match = re.search(r'CHOSEN SUSPICION:\s*(\d+)', response, re.IGNORECASE)
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n\n|\n[A-Z]|\Z)', response, re.IGNORECASE | re.DOTALL)
        
        if chosen_match:
            try:
                choice_num = int(chosen_match.group(1))
                if 1 <= choice_num <= len(suspicions):
                    chosen_suspicion = suspicions[choice_num - 1]
            except ValueError:
                pass
        
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        # Fallback: look for any number if structured format failed
        if chosen_suspicion is None:
            number_match = re.search(r'\b(\d+)\b', response)
            if number_match:
                try:
                    choice_num = int(number_match.group(1))
                    if 1 <= choice_num <= len(suspicions):
                        chosen_suspicion = suspicions[choice_num - 1]
                except ValueError:
                    pass
        
        # Final fallback for chosen suspicion
        if chosen_suspicion is None:
            chosen_suspicion = suspicions[0] if suspicions else "Unknown"
        
        # If no structured reasoning found, try to extract any reasoning text
        if not reasoning:
            # Look for reasoning patterns
            reasoning_patterns = [
                r'because\s+(.+?)(?=\n|$)',
                r'since\s+(.+?)(?=\n|$)', 
                r'due to\s+(.+?)(?=\n|$)',
                r'based on\s+(.+?)(?=\n|$)',
                r'given\s+(.+?)(?=\n|$)',
                r'shows\s+(.+?)(?=\n|$)',
                r'indicates\s+(.+?)(?=\n|$)',
                r'consistent with\s+(.+?)(?=\n|$)'
            ]
            
            for pattern in reasoning_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    reasoning = match.group(1).strip()
                    break
            
            # If still no reasoning, use the whole response as reasoning context
            if not reasoning and len(response.strip()) > 50:
                reasoning = response.strip()
        
        return chosen_suspicion, reasoning
    
    def progressive_iterative_reasoning(self, sample: Dict, chosen_suspicion: str, 
                                      all_suspicions: List[str], max_steps: int = 5) -> Dict:
        """REFACTORED: Progressive reasoning that builds on Stage 3 choice instead of starting over"""
        
        # Get the complete patient summary for Stage 4 reasoning
        patient_summary = self.create_patient_data_summary(sample, 6)
        
        # Map suspicion to category for flowchart navigation
        suspected_category = self.map_suspicion_to_category(chosen_suspicion)
        
        # If no mapping found, try direct category match
        if suspected_category is None and chosen_suspicion in self.flowchart_categories:
            suspected_category = chosen_suspicion
        
        if suspected_category:
            # Use progressive flowchart reasoning that builds on Stage 3
            reasoning_result = self.progressive_flowchart_reasoning(
                sample, chosen_suspicion, suspected_category, patient_summary, max_steps
            )
        else:
            # Fallback: direct match without flowcharts
            matched_diagnosis = self.find_best_match(chosen_suspicion)
            reasoning_result = {
                'final_diagnosis': matched_diagnosis,
                'reasoning_trace': [{
                    'step': 1, 
                    'action': 'direct_match', 
                    'chosen_suspicion': chosen_suspicion,
                    'matched_diagnosis': matched_diagnosis,
                    'response': f'Stage 3 chose "{chosen_suspicion}", matched to diagnosis "{matched_diagnosis}"'
                }],
                'reasoning_steps': 1,
                'reasoning_successful': True
            }
        
        return reasoning_result
    
    def progressive_flowchart_reasoning(self, sample: Dict, chosen_suspicion: str, 
                                     category: str, patient_summary: str, max_steps: int = 5) -> Dict:
        """NEW: Progressive reasoning that builds on Stage 3 choice with flowchart guidance"""
        
        reasoning_trace = []
        
        # Load flowchart for the category
        try:
            flowchart_data = load_flowchart_content(category, self.flowchart_dir)
            flowchart_structure = get_flowchart_structure(flowchart_data)
            flowchart_knowledge = get_flowchart_knowledge(flowchart_data)
        except Exception as e:
            print(f"Warning: Could not load flowchart for {category}: {e}")
            # Fallback to direct matching
            matched_diagnosis = self.find_best_match(chosen_suspicion)
            return {
                'final_diagnosis': matched_diagnosis,
                'reasoning_trace': [{
                    'step': 1, 
                    'action': 'flowchart_unavailable',
                    'response': f'Flowchart for {category} unavailable, using direct match: {matched_diagnosis}'
                }],
                'reasoning_steps': 1,
                'reasoning_successful': True
            }
        
        # Step 1: Build on Stage 3 reasoning with flowchart guidance
        stage4_initial_prompt = self.create_stage4_initial_prompt(
            chosen_suspicion, category, patient_summary, flowchart_knowledge
        )
        stage4_initial_response = self.query_llm(stage4_initial_prompt, max_tokens=800)
        
        # Parse the initial Stage 4 response for next steps
        next_step_info = self.parse_stage4_initial_response(stage4_initial_response, flowchart_structure)
        
        reasoning_trace.append({
            'step': 1,
            'category': category,
            'chosen_suspicion': chosen_suspicion,
            'current_node': next_step_info.get('current_node', chosen_suspicion),
            'action': 'stage4_initial_reasoning',
            'prompt': stage4_initial_prompt,
            'response': stage4_initial_response,
            'reasoning_type': 'building_on_stage3_choice'
        })
        
        # Continue with flowchart-guided reasoning if needed
        current_node = next_step_info.get('current_node', chosen_suspicion)
        current_step = 2
        
        # If we already have a final diagnosis from Stage 4 initial reasoning, use it
        if next_step_info.get('final_diagnosis'):
            final_diagnosis = next_step_info['final_diagnosis']
        else:
            # Continue with iterative flowchart reasoning
            final_diagnosis = current_node
            
            while current_step <= max_steps:
                # Check if current node is a final diagnosis
                if is_leaf_diagnosis(flowchart_structure, current_node):
                    final_diagnosis = current_node
                    break
                
                # Get possible next steps from flowchart
                children = get_flowchart_children(flowchart_structure, current_node)
                if not children:
                    final_diagnosis = current_node
                    break
                
                # Create reasoning step prompt that builds on previous reasoning
                step_prompt = self.create_progressive_step_prompt(
                    current_step, current_node, children, patient_summary, 
                    reasoning_trace[-1]['response']  # Previous reasoning
                )
                
                step_response = self.query_llm(step_prompt, max_tokens=800)
                step_result = extract_reasoning_choice(step_response, children)
                chosen_option = step_result['chosen_option']
                
                reasoning_trace.append({
                    'step': current_step,
                    'category': category,
                    'current_node': current_node,
                    'available_options': children,
                    'chosen_option': chosen_option,
                    'action': 'progressive_reasoning_step',
                    'prompt': step_prompt,
                    'response': step_response,
                    'evidence_matching': step_result.get('evidence_matching', ''),
                    'comparative_analysis': step_result.get('comparative_analysis', ''),
                    'rationale': step_result.get('rationale', '')
                })
                
                current_node = chosen_option
                current_step += 1
            
            final_diagnosis = current_node
        
        # Match final diagnosis against possible diagnoses
        matched_diagnosis = self.find_best_match(final_diagnosis)
        
        return {
            'final_diagnosis': matched_diagnosis,
            'reasoning_trace': reasoning_trace,
            'reasoning_steps': len(reasoning_trace),
            'reasoning_successful': bool(matched_diagnosis),
            'category_used': category
        }
    
    def create_stage4_initial_prompt(self, chosen_suspicion: str, category: str, 
                                   patient_summary: str, flowchart_knowledge: Dict) -> str:
        """Create prompt for Stage 4 initial reasoning that builds on Stage 3 choice"""
        
        prompt = f"""You are a medical expert in Stage 4 of progressive diagnostic reasoning. In Stage 3, you chose "{chosen_suspicion}" as the most likely diagnosis based on the complete clinical information.

**Stage 3 Choice:** {chosen_suspicion}
**Diagnostic Category:** {category}

**Complete Patient Clinical Information:**
{patient_summary}

**Relevant Medical Knowledge for {category}:**"""

        # Add flowchart knowledge
        if flowchart_knowledge:
            for knowledge_key, knowledge_content in flowchart_knowledge.items():
                if isinstance(knowledge_content, dict):
                    prompt += f"\n**{knowledge_key}:**\n"
                    for sub_key, sub_content in knowledge_content.items():
                        prompt += f"• {sub_key}: {sub_content}\n"
                elif isinstance(knowledge_content, str):
                    prompt += f"\n**{knowledge_key}:**\n{knowledge_content}\n"
        
        # CRITICAL FIX: Add the possible diagnoses list to constrain LLM choices
        prompt += f"""

**Possible Primary Discharge Diagnoses:**"""
        for i, diagnosis in enumerate(self.possible_diagnoses, 1):
            prompt += f"\n{i}. {diagnosis}"
        
        prompt += f"""

**Task:** Build on your Stage 3 choice of "{chosen_suspicion}" by using the medical knowledge above to:
1. Confirm or refine your diagnostic thinking
2. Identify the most specific diagnosis within the {category} category from the possible diagnoses list
3. Explain how the clinical findings support your final diagnosis

**Instructions:**
• Start with your Stage 3 choice of "{chosen_suspicion}" as the foundation
• Use the clinical information to support or refine this choice
• Apply the {category} medical knowledge to reach a specific final diagnosis
• IMPORTANT: Your final diagnosis MUST be selected from the possible diagnoses list above
• Provide detailed medical reasoning for your final diagnosis

**Format:**
**BUILDING ON STAGE 3:** [Explain how clinical findings support your choice of {chosen_suspicion}]
**REFINED ANALYSIS:** [Apply medical knowledge to refine the diagnosis]
**FINAL DIAGNOSIS:** [Exact diagnosis name from the possible diagnoses list above]
**REASONING:** [Complete medical reasoning for the final diagnosis]

**Stage 4 Analysis:**"""

        return prompt
    
    def parse_stage4_initial_response(self, response: str, flowchart_structure: Dict) -> Dict:
        """Parse Stage 4 initial response to extract next steps"""
        
        import re
        
        result = {}
        
        # Try to extract final diagnosis
        diagnosis_match = re.search(r'FINAL DIAGNOSIS:\s*(.+?)(?=\n\*\*|\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if diagnosis_match:
            diagnosis = diagnosis_match.group(1).strip()
            # Clean up markdown formatting
            diagnosis = re.sub(r'^\*+', '', diagnosis)  # Remove leading asterisks
            diagnosis = re.sub(r'\*+$', '', diagnosis)  # Remove trailing asterisks
            diagnosis = diagnosis.strip()
            result['final_diagnosis'] = diagnosis
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n\*\*|\Z)', response, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            # Clean up markdown formatting
            reasoning = re.sub(r'^\*+', '', reasoning)
            reasoning = re.sub(r'\*+$', '', reasoning)
            reasoning = reasoning.strip()
            result['reasoning'] = reasoning
        
        # Determine current node (use final diagnosis if available)
        if result.get('final_diagnosis'):
            result['current_node'] = result['final_diagnosis']
        else:
            # Fallback to extracting from response
            result['current_node'] = 'Suspected Diagnosis'
        
        return result
    
    def create_progressive_step_prompt(self, step: int, current_node: str, children: List[str], 
                                     patient_summary: str, previous_reasoning: str) -> str:
        """Create prompt for progressive reasoning steps that build on previous reasoning"""
        
        prompt = f"""You are continuing progressive diagnostic reasoning in Step {step}.

**Previous Reasoning:**
{previous_reasoning}

**Current Diagnostic Consideration:** {current_node}

**Complete Patient Clinical Information:**
{patient_summary}

**Available Next Steps:**"""
        
        for i, child in enumerate(children, 1):
            prompt += f"\n{i}. {child}"
        
        prompt += f"""

**Task:** Based on your previous reasoning and the complete clinical information, choose the most appropriate next step.

**Instructions:**
• Build on your previous reasoning above
• Compare each option against the patient's clinical findings
• Choose the option most consistent with the evidence
• Provide detailed medical reasoning for your choice

**Format:**
**EVIDENCE MATCHING:** [How patient findings match each option]
**COMPARATIVE ANALYSIS:** [Why chosen option is better than alternatives]
**CHOSEN OPTION:** [Number] - [Option name]
**RATIONALE:** [Complete reasoning for this choice]

**Step {step} Analysis:**"""

        return prompt
    
    def map_suspicion_to_category(self, suspicion: str) -> str:
        """Map a specific suspicion to a disease category for flowchart navigation"""
        
        suspicion_lower = suspicion.lower()
        
        # CRITICAL FIX: Check for exact matches first before keyword matching
        # This prevents "Heart Failure" from mapping to "Acute Coronary Syndrome"
        for flowchart_category in self.flowchart_categories:
            if suspicion_lower == flowchart_category.lower():
                return flowchart_category
        
        # CRITICAL FIX: Handle tuberculosis specifically - it should map to Pneumonia for respiratory symptoms
        # since both tuberculosis and bacterial pneumonia are respiratory infections
        if 'tuberculosis' in suspicion_lower or 'tb' == suspicion_lower:
            # Look for Pneumonia category first (broader respiratory infectious disease category)
            for flowchart_category in self.flowchart_categories:
                if 'pneumonia' in flowchart_category.lower():
                    return flowchart_category
            # Fallback to any respiratory category
            for flowchart_category in self.flowchart_categories:
                if any(keyword in flowchart_category.lower() for keyword in ['respiratory', 'lung', 'pulmonary']):
                    return flowchart_category
        
        # CRITICAL FIX: Handle pneumonia suspicions - they should map directly to Pneumonia category
        if 'pneumonia' in suspicion_lower:
            for flowchart_category in self.flowchart_categories:
                if 'pneumonia' in flowchart_category.lower():
                    return flowchart_category
        
        # Enhanced mapping with better coverage for respiratory diseases
        category_keywords = {
            'cardiovascular': ['heart', 'cardiac', 'myocardial', 'coronary', 'angina', 'infarction', 'arrhythmia', 'hypertension'],
            'respiratory': ['lung', 'pulmonary', 'asthma', 'copd', 'respiratory', 'bronch'],
            'gastrointestinal': ['gastric', 'intestinal', 'bowel', 'stomach', 'liver', 'pancreatic', 'gallbladder'],
            'neurological': ['stroke', 'seizure', 'neurologic', 'brain', 'headache', 'migraine'],
            'endocrine': ['diabetes', 'thyroid', 'hormone', 'endocrine', 'metabolic'],
            'infectious': ['infection', 'sepsis', 'bacterial', 'viral'],
            'renal': ['kidney', 'renal', 'urinary', 'nephro'],
            'hematologic': ['anemia', 'bleeding', 'hematologic', 'blood'],
        }
        
        # Standard mapping for other suspicions (only after exact matching fails)
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in suspicion_lower:
                    # Find the actual category name in our flowchart categories
                    for flowchart_category in self.flowchart_categories:
                        if category.lower() in flowchart_category.lower():
                            return flowchart_category
                    # If exact match not found, try partial matching
                    for flowchart_category in self.flowchart_categories:
                        if any(kw in flowchart_category.lower() for kw in keywords):
                            return flowchart_category
        
        return None

    def extract_tests_from_recommendations(self, recommended_tests: str) -> List[str]:
        """Extract individual test names from LLM recommendations"""
        
        # Common test abbreviations and their full names
        test_mappings = {
            'cbc': 'complete blood count',
            'bmp': 'basic metabolic panel',
            'cmp': 'comprehensive metabolic panel',
            'lipid panel': 'lipid profile',
            'liver function': 'liver function tests',
            'lft': 'liver function tests',
            'kidney function': 'renal function tests',
            'rft': 'renal function tests',
            'cardiac enzymes': 'troponin',
            'troponins': 'troponin',
            'ekg': 'electrocardiogram',
            'ecg': 'electrocardiogram',
            'chest x-ray': 'chest xray',
            'cxr': 'chest xray',
            'ct scan': 'ct',
            'mri scan': 'mri',
            'urinalysis': 'urine analysis',
            'ua': 'urine analysis',
            'blood pressure': 'bp',
            'heart rate': 'pulse',
            'respiratory rate': 'breathing',
            'temperature': 'temp',
            'oxygen saturation': 'spo2',
            'pulse oximetry': 'spo2'
        }
        
        import re
        
        # Convert to lowercase for processing
        text = recommended_tests.lower()
        
        # Extract potential test names
        extracted_tests = set()
        
        # Look for common patterns
        patterns = [
            r'\b(?:order|obtain|check|perform|do)\s+([a-zA-Z0-9\s]+?)(?:\s+to|\s+for|\s+in|\.|\,|$)',
            r'\b(cbc|bmp|cmp|troponin|ekg|ecg|chest x-ray|cxr|ct|mri|urinalysis|ua)\b',
            r'\b(complete blood count|basic metabolic|comprehensive metabolic|liver function|cardiac enzymes)\b',
            r'\b(blood pressure|heart rate|respiratory rate|temperature|pulse|breathing)\b',
            r'\b([a-zA-Z]+\s+(?:levels?|test|exam|study|scan|panel|profile|analysis))\b',
            r'\b(?:physical exam|examination):\s*([a-zA-Z0-9\s,]+?)(?:\n|$)',
            r'\b(?:lab|laboratory):\s*([a-zA-Z0-9\s,]+?)(?:\n|$)',
            r'\b(?:imaging|radiology):\s*([a-zA-Z0-9\s,]+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                test_name = match.group(1).strip()
                if test_name and len(test_name) > 2:
                    # Clean up the test name
                    test_name = re.sub(r'[^\w\s]', '', test_name)
                    test_name = ' '.join(test_name.split())
                    
                    # Apply mappings
                    normalized = test_mappings.get(test_name, test_name)
                    extracted_tests.add(normalized)
        
        # Also look for bullet points and numbered lists
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^[•\-\*\d+\.]\s*', line):
                # Remove bullet/number prefix
                clean_line = re.sub(r'^[•\-\*\d+\.]\s*', '', line)
                clean_line = re.sub(r'[^\w\s]', '', clean_line)
                clean_line = ' '.join(clean_line.split())
                
                if clean_line and len(clean_line) > 2:
                    # Apply mappings
                    normalized = test_mappings.get(clean_line, clean_line)
                    extracted_tests.add(normalized)
        
        return list(extracted_tests)
    
    def extract_tests_from_clinical_data(self, sample: Dict) -> List[str]:
        """Extract test names from actual clinical data (inputs 5 and 6)"""
        
        # Test indicators in clinical text
        test_indicators = {
            'complete blood count': ['cbc', 'complete blood count', 'blood count'],
            'troponin': ['troponin', 'cardiac enzymes', 'troponin i', 'troponin t'],
            'electrocardiogram': ['ekg', 'ecg', 'electrocardiogram'],
            'chest xray': ['chest x-ray', 'cxr', 'chest xray', 'chest radiograph'],
            'ct': ['ct scan', 'computed tomography', 'ct chest', 'ct abdomen'],
            'mri': ['mri', 'magnetic resonance', 'mri brain'],
            'urine analysis': ['urinalysis', 'ua', 'urine analysis', 'urine test'],
            'liver function tests': ['lft', 'liver function', 'alt', 'ast', 'bilirubin'],
            'renal function tests': ['creatinine', 'bun', 'kidney function', 'renal function'],
            'lipid profile': ['lipid panel', 'cholesterol', 'triglycerides', 'hdl', 'ldl'],
            'glucose': ['glucose', 'blood sugar', 'blood glucose'],
            'bp': ['blood pressure', 'bp', 'hypertension', 'hypotension'],
            'pulse': ['heart rate', 'pulse', 'hr'],
            'breathing': ['respiratory rate', 'rr', 'breathing'],
            'temp': ['temperature', 'fever', 'temp'],
            'spo2': ['oxygen saturation', 'spo2', 'pulse oximetry', 'o2 sat']
        }
        
        import re
        
        # Combine physical exam and lab results
        clinical_text = ""
        if 'input5' in sample:  # Physical Examination
            clinical_text += sample['input5'].lower() + " "
        if 'input6' in sample:  # Laboratory Results
            clinical_text += sample['input6'].lower() + " "
        
        # Find tests that were actually performed
        performed_tests = set()
        
        for test_name, indicators in test_indicators.items():
            for indicator in indicators:
                # Look for the indicator in the clinical text
                if re.search(r'\b' + re.escape(indicator) + r'\b', clinical_text):
                    performed_tests.add(test_name)
                    break  # Found one indicator for this test
        
        # Look for numerical values that indicate lab results
        lab_patterns = [
            r'\b(?:wbc|white blood cell)\s*:?\s*[\d\.,]+',
            r'\b(?:hgb|hemoglobin)\s*:?\s*[\d\.,]+',
            r'\b(?:plt|platelet)\s*:?\s*[\d\.,]+',
            r'\b(?:sodium|na)\s*:?\s*[\d\.,]+',
            r'\b(?:potassium|k)\s*:?\s*[\d\.,]+',
            r'\b(?:chloride|cl)\s*:?\s*[\d\.,]+',
            r'\b(?:co2|bicarbonate)\s*:?\s*[\d\.,]+',
            r'\b(?:bun|urea)\s*:?\s*[\d\.,]+',
            r'\b(?:creatinine|cr)\s*:?\s*[\d\.,]+',
            r'\b(?:glucose|glu)\s*:?\s*[\d\.,]+',
        ]
        
        for pattern in lab_patterns:
            if re.search(pattern, clinical_text):
                # Determine which test this result belongs to
                if 'wbc' in pattern or 'hemoglobin' in pattern or 'platelet' in pattern:
                    performed_tests.add('complete blood count')
                elif 'sodium' in pattern or 'potassium' in pattern or 'chloride' in pattern or 'co2' in pattern:
                    performed_tests.add('basic metabolic panel')
                elif 'bun' in pattern or 'creatinine' in pattern:
                    performed_tests.add('renal function tests')
                elif 'glucose' in pattern:
                    performed_tests.add('glucose')
        
        return list(performed_tests)
    
    def calculate_test_overlap_metrics(self, recommended_tests: str, sample: Dict) -> Dict:
        """Calculate overlap metrics between recommended and actual tests"""
        
        # Use LLM-based approach if enabled, otherwise use regex-based approach
        if self.llm_test_overlap:
            try:
                return self.calculate_test_overlap_metrics_llm(recommended_tests, sample)
            except Exception as e:
                print(f"Error in LLM-based test overlap calculation, falling back to original method: {e}")
                return self.calculate_test_overlap_metrics_original(recommended_tests, sample)
        else:
            return self.calculate_test_overlap_metrics_original(recommended_tests, sample)
    
    def calculate_test_overlap_metrics_original(self, recommended_tests: str, sample: Dict) -> Dict:
        """Calculate overlap metrics between recommended and actual tests (original regex-based method)"""
        
        # Extract test lists
        recommended = set(self.extract_tests_from_recommendations_original(recommended_tests))
        actual = set(self.extract_tests_from_clinical_data_original(sample))
        
        # Remove empty strings
        recommended = {t for t in recommended if t.strip()}
        actual = {t for t in actual if t.strip()}
        
        # Calculate metrics
        intersection = recommended & actual
        union = recommended | actual
        
        # Basic counts
        num_recommended = len(recommended)
        num_actual = len(actual)
        num_overlap = len(intersection)
        
        # Calculate metrics (handle division by zero)
        precision = num_overlap / num_recommended if num_recommended > 0 else 0
        recall = num_overlap / num_actual if num_actual > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        jaccard_index = num_overlap / len(union) if len(union) > 0 else 0
        
        # Additional metrics
        unnecessary_tests = recommended - actual  # Tests recommended but not done
        missed_tests = actual - recommended       # Tests done but not recommended
        
        unnecessary_rate = len(unnecessary_tests) / num_recommended if num_recommended > 0 else 0
        missed_rate = len(missed_tests) / num_actual if num_actual > 0 else 0
        
        return {
            'test_overlap_precision': precision,
            'test_overlap_recall': recall,
            'test_overlap_f1': f1_score,
            'test_overlap_jaccard': jaccard_index,
            'tests_recommended_count': num_recommended,
            'tests_actual_count': num_actual,
            'tests_overlap_count': num_overlap,
            'unnecessary_tests_count': len(unnecessary_tests),
            'missed_tests_count': len(missed_tests),
            'unnecessary_tests_rate': unnecessary_rate,
            'missed_tests_rate': missed_rate,
            'recommended_tests_list': list(recommended),
            'actual_tests_list': list(actual),
            'overlap_tests_list': list(intersection),
            'unnecessary_tests_list': list(unnecessary_tests),
            'missed_tests_list': list(missed_tests)
        }
    
    def extract_tests_from_recommendations_llm(self, recommended_tests: str) -> List[str]:
        """Extract individual test names from LLM recommendations using LLM as judge"""
        
        if not recommended_tests.strip():
            return []
        
        extraction_prompt = f"""You are a medical expert tasked with extracting specific test names from clinical recommendations.

**Clinical Recommendations:**
{recommended_tests}

**Task:** Extract all specific medical tests, laboratory tests, imaging studies, and physical examination components mentioned in the recommendations above.

**Instructions:**
• List each test as a separate item
• Use standard medical terminology (e.g., "Complete Blood Count" not "blood work")
• Include both specific tests (e.g., "Troponin I") and general categories (e.g., "Cardiac Enzymes")
• Include physical exam components (e.g., "Blood Pressure", "Heart Rate")
• Include imaging studies (e.g., "Chest X-ray", "ECG")
• Do not include general phrases like "monitor" or "assess"

**Format:** List one test per line, like:
- Complete Blood Count
- Electrocardiogram
- Chest X-ray
- Blood Pressure
- Troponin

**Tests Mentioned:**"""
        
        try:
            response = self.query_llm(extraction_prompt, max_tokens=300)
            
            # Parse the response to extract test names
            tests = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                # Remove bullet points, dashes, numbers
                clean_line = re.sub(r'^[-•\*\d+\.]\s*', '', line).strip()
                if clean_line and len(clean_line) > 2 and not clean_line.lower().startswith('test'):
                    tests.append(clean_line)
            
            return tests[:15]  # Limit to reasonable number
            
        except Exception as e:
            print(f"Error in LLM test extraction: {e}")
            # Fallback to original method
            return self.extract_tests_from_recommendations_original(recommended_tests)
    
    def extract_tests_from_clinical_data_llm(self, sample: Dict) -> List[str]:
        """Extract test names from actual clinical data using LLM as judge"""
        
        # Combine physical exam and lab results
        clinical_text = ""
        if 'input5' in sample:  # Physical Examination
            clinical_text += f"Physical Examination: {sample['input5']}\n\n"
        if 'input6' in sample:  # Laboratory Results
            clinical_text += f"Laboratory Results: {sample['input6']}\n\n"
        
        if not clinical_text.strip():
            return []
        
        extraction_prompt = f"""You are a medical expert tasked with identifying which specific tests were actually performed based on clinical documentation.

**Clinical Documentation:**
{clinical_text}

**Task:** Identify all medical tests, laboratory tests, imaging studies, and physical examination components that were actually performed or measured, as evidenced by the documentation above.

**Instructions:**
• Only include tests that were clearly performed (have results, measurements, or findings)
• Use standard medical terminology
• Include physical exam components with measurements (e.g., "Blood Pressure", "Heart Rate")
• Include lab tests with values (e.g., "Complete Blood Count", "Troponin")
• Include imaging studies mentioned (e.g., "Chest X-ray", "ECG")
• Do not include tests that were only planned or recommended

**Format:** List one test per line, like:
- Blood Pressure
- Complete Blood Count
- Electrocardiogram
- Chest X-ray

**Tests Actually Performed:**"""
        
        try:
            response = self.query_llm(extraction_prompt, max_tokens=300)
            
            # Parse the response to extract test names
            tests = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                # Remove bullet points, dashes, numbers
                clean_line = re.sub(r'^[-•\*\d+\.]\s*', '', line).strip()
                if clean_line and len(clean_line) > 2 and not clean_line.lower().startswith('test'):
                    tests.append(clean_line)
            
            return tests[:15]  # Limit to reasonable number
            
        except Exception as e:
            print(f"Error in LLM clinical test extraction: {e}")
            # Fallback to original method
            return self.extract_tests_from_clinical_data_original(sample)
    
    def judge_test_equivalence_llm(self, recommended_test: str, actual_test: str) -> bool:
        """Use LLM to judge if recommended test and actual test are equivalent"""
        
        judge_prompt = f"""You are a medical expert evaluating whether two medical tests refer to the same or equivalent procedures.

**Recommended Test:** "{recommended_test}"
**Actual Test:** "{actual_test}"

**Task:** Determine if these two tests are the same, equivalent, or measure the same thing.

**Consider equivalent:**
• Different names for the same test (e.g., "ECG" and "Electrocardiogram")
• General category and specific test (e.g., "Cardiac Enzymes" and "Troponin I")
• Different phrasing for same measurement (e.g., "Blood Pressure" and "BP")
• Synonymous medical terms (e.g., "CBC" and "Complete Blood Count")

**Consider NOT equivalent:**
• Tests that measure completely different things
• Different body systems or organs

**Question:** Are these two tests equivalent or the same?

**Response:** Answer ONLY "YES" if they are equivalent/same, or "NO" if they are different.

**Answer:**"""
        
        try:
            response = self.query_llm(judge_prompt, max_tokens=10, temperature=0.0)
            return response.strip().upper() == "YES"
        except Exception as e:
            print(f"Error in LLM test equivalence judging: {e}")
            # Fallback to simple string matching
            return recommended_test.lower() in actual_test.lower() or actual_test.lower() in recommended_test.lower()
    
    def calculate_test_overlap_metrics_llm(self, recommended_tests: str, sample: Dict) -> Dict:
        """Calculate overlap metrics between recommended and actual tests using LLM judge"""
        
        # Extract test lists using LLM
        recommended = self.extract_tests_from_recommendations_llm(recommended_tests)
        actual = self.extract_tests_from_clinical_data_llm(sample)
        
        if not recommended and not actual:
            # Both empty
            return {
                'test_overlap_precision': 1.0,
                'test_overlap_recall': 1.0,
                'test_overlap_f1': 1.0,
                'test_overlap_jaccard': 1.0,
                'tests_recommended_count': 0,
                'tests_actual_count': 0,
                'tests_overlap_count': 0,
                'unnecessary_tests_count': 0,
                'missed_tests_count': 0,
                'unnecessary_tests_rate': 0.0,
                'missed_tests_rate': 0.0,
                'recommended_tests_list': [],
                'actual_tests_list': [],
                'overlap_tests_list': [],
                'unnecessary_tests_list': [],
                'missed_tests_list': []
            }
        
        # Use LLM to judge equivalence for each pair
        overlap_tests = []
        unnecessary_tests = []
        missed_tests = []
        
        # Find overlapping tests
        for rec_test in recommended:
            found_match = False
            for act_test in actual:
                if self.judge_test_equivalence_llm(rec_test, act_test):
                    overlap_tests.append(f"{rec_test} ≈ {act_test}")
                    found_match = True
                    break
            if not found_match:
                unnecessary_tests.append(rec_test)
        
        # Find missed tests (in actual but not recommended)
        for act_test in actual:
            found_match = False
            for rec_test in recommended:
                if self.judge_test_equivalence_llm(rec_test, act_test):
                    found_match = True
                    break
            if not found_match:
                missed_tests.append(act_test)
        
        # Calculate metrics
        num_recommended = len(recommended)
        num_actual = len(actual)
        num_overlap = len(overlap_tests)
        
        # Calculate metrics (handle division by zero)
        precision = num_overlap / num_recommended if num_recommended > 0 else 0
        recall = num_overlap / num_actual if num_actual > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # For Jaccard, we need unique count (since overlap_tests has pairs)
        total_unique_tests = num_recommended + num_actual - num_overlap
        jaccard_index = num_overlap / total_unique_tests if total_unique_tests > 0 else 0
        
        unnecessary_rate = len(unnecessary_tests) / num_recommended if num_recommended > 0 else 0
        missed_rate = len(missed_tests) / num_actual if num_actual > 0 else 0
        
        return {
            'test_overlap_precision': precision,
            'test_overlap_recall': recall,
            'test_overlap_f1': f1_score,
            'test_overlap_jaccard': jaccard_index,
            'tests_recommended_count': num_recommended,
            'tests_actual_count': num_actual,
            'tests_overlap_count': num_overlap,
            'unnecessary_tests_count': len(unnecessary_tests),
            'missed_tests_count': len(missed_tests),
            'unnecessary_tests_rate': unnecessary_rate,
            'missed_tests_rate': missed_rate,
            'recommended_tests_list': recommended,
            'actual_tests_list': actual,
            'overlap_tests_list': overlap_tests,
            'unnecessary_tests_list': unnecessary_tests,
            'missed_tests_list': missed_tests
        }
    
    def extract_tests_from_recommendations_original(self, recommended_tests: str) -> List[str]:
        """Extract individual test names from LLM recommendations"""
        
        # Common test abbreviations and their full names
        test_mappings = {
            'cbc': 'complete blood count',
            'bmp': 'basic metabolic panel',
            'cmp': 'comprehensive metabolic panel',
            'lipid panel': 'lipid profile',
            'liver function': 'liver function tests',
            'lft': 'liver function tests',
            'kidney function': 'renal function tests',
            'rft': 'renal function tests',
            'cardiac enzymes': 'troponin',
            'troponins': 'troponin',
            'ekg': 'electrocardiogram',
            'ecg': 'electrocardiogram',
            'chest x-ray': 'chest xray',
            'cxr': 'chest xray',
            'ct scan': 'ct',
            'mri scan': 'mri',
            'urinalysis': 'urine analysis',
            'ua': 'urine analysis',
            'blood pressure': 'bp',
            'heart rate': 'pulse',
            'respiratory rate': 'breathing',
            'temperature': 'temp',
            'oxygen saturation': 'spo2',
            'pulse oximetry': 'spo2'
        }
        
        import re
        
        # Convert to lowercase for processing
        text = recommended_tests.lower()
        
        # Extract potential test names
        extracted_tests = set()
        
        # Look for common patterns
        patterns = [
            r'\b(?:order|obtain|check|perform|do)\s+([a-zA-Z0-9\s]+?)(?:\s+to|\s+for|\s+in|\.|\,|$)',
            r'\b(cbc|bmp|cmp|troponin|ekg|ecg|chest x-ray|cxr|ct|mri|urinalysis|ua)\b',
            r'\b(complete blood count|basic metabolic|comprehensive metabolic|liver function|cardiac enzymes)\b',
            r'\b(blood pressure|heart rate|respiratory rate|temperature|pulse|breathing)\b',
            r'\b([a-zA-Z]+\s+(?:levels?|test|exam|study|scan|panel|profile|analysis))\b',
            r'\b(?:physical exam|examination):\s*([a-zA-Z0-9\s,]+?)(?:\n|$)',
            r'\b(?:lab|laboratory):\s*([a-zA-Z0-9\s,]+?)(?:\n|$)',
            r'\b(?:imaging|radiology):\s*([a-zA-Z0-9\s,]+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                test_name = match.group(1).strip()
                if test_name and len(test_name) > 2:
                    # Clean up the test name
                    test_name = re.sub(r'[^\w\s]', '', test_name)
                    test_name = ' '.join(test_name.split())
                    
                    # Apply mappings
                    normalized = test_mappings.get(test_name, test_name)
                    extracted_tests.add(normalized)
        
        # Also look for bullet points and numbered lists
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^[•\-\*\d+\.]\s*', line):
                # Remove bullet/number prefix
                clean_line = re.sub(r'^[•\-\*\d+\.]\s*', '', line)
                clean_line = re.sub(r'[^\w\s]', '', clean_line)
                clean_line = ' '.join(clean_line.split())
                
                if clean_line and len(clean_line) > 2:
                    # Apply mappings
                    normalized = test_mappings.get(clean_line, clean_line)
                    extracted_tests.add(normalized)
        
        return list(extracted_tests)
    
    def extract_tests_from_clinical_data_original(self, sample: Dict) -> List[str]:
        """Extract test names from actual clinical data (inputs 5 and 6)"""
        
        # Test indicators in clinical text
        test_indicators = {
            'complete blood count': ['cbc', 'complete blood count', 'blood count'],
            'troponin': ['troponin', 'cardiac enzymes', 'troponin i', 'troponin t'],
            'electrocardiogram': ['ekg', 'ecg', 'electrocardiogram'],
            'chest xray': ['chest x-ray', 'cxr', 'chest xray', 'chest radiograph'],
            'ct': ['ct scan', 'computed tomography', 'ct chest', 'ct abdomen'],
            'mri': ['mri', 'magnetic resonance', 'mri brain'],
            'urine analysis': ['urinalysis', 'ua', 'urine analysis', 'urine test'],
            'liver function tests': ['lft', 'liver function', 'alt', 'ast', 'bilirubin'],
            'renal function tests': ['creatinine', 'bun', 'kidney function', 'renal function'],
            'lipid profile': ['lipid panel', 'cholesterol', 'triglycerides', 'hdl', 'ldl'],
            'glucose': ['glucose', 'blood sugar', 'blood glucose'],
            'bp': ['blood pressure', 'bp', 'hypertension', 'hypotension'],
            'pulse': ['heart rate', 'pulse', 'hr'],
            'breathing': ['respiratory rate', 'rr', 'breathing'],
            'temp': ['temperature', 'fever', 'temp'],
            'spo2': ['oxygen saturation', 'spo2', 'pulse oximetry', 'o2 sat']
        }
        
        import re
        
        # Combine physical exam and lab results
        clinical_text = ""
        if 'input5' in sample:  # Physical Examination
            clinical_text += sample['input5'].lower() + " "
        if 'input6' in sample:  # Laboratory Results
            clinical_text += sample['input6'].lower() + " "
        
        # Find tests that were actually performed
        performed_tests = set()
        
        for test_name, indicators in test_indicators.items():
            for indicator in indicators:
                # Look for the indicator in the clinical text
                if re.search(r'\b' + re.escape(indicator) + r'\b', clinical_text):
                    performed_tests.add(test_name)
                    break  # Found one indicator for this test
        
        # Look for numerical values that indicate lab results
        lab_patterns = [
            r'\b(?:wbc|white blood cell)\s*:?\s*[\d\.,]+',
            r'\b(?:hgb|hemoglobin)\s*:?\s*[\d\.,]+',
            r'\b(?:plt|platelet)\s*:?\s*[\d\.,]+',
            r'\b(?:sodium|na)\s*:?\s*[\d\.,]+',
            r'\b(?:potassium|k)\s*:?\s*[\d\.,]+',
            r'\b(?:chloride|cl)\s*:?\s*[\d\.,]+',
            r'\b(?:co2|bicarbonate)\s*:?\s*[\d\.,]+',
            r'\b(?:bun|urea)\s*:?\s*[\d\.,]+',
            r'\b(?:creatinine|cr)\s*:?\s*[\d\.,]+',
            r'\b(?:glucose|glu)\s*:?\s*[\d\.,]+',
        ]
        
        for pattern in lab_patterns:
            if re.search(pattern, clinical_text):
                # Determine which test this result belongs to
                if 'wbc' in pattern or 'hemoglobin' in pattern or 'platelet' in pattern:
                    performed_tests.add('complete blood count')
                elif 'sodium' in pattern or 'potassium' in pattern or 'chloride' in pattern or 'co2' in pattern:
                    performed_tests.add('basic metabolic panel')
                elif 'bun' in pattern or 'creatinine' in pattern:
                    performed_tests.add('renal function tests')
                elif 'glucose' in pattern:
                    performed_tests.add('glucose')
        
        return list(performed_tests)
    
    def create_initial_reasoning_prompt(self, patient_summary: str, category: str, 
                                      flowchart_knowledge: Dict, current_node: str) -> str:
        """Create prompt for initial reasoning step that analyzes clinical evidence against flowchart knowledge"""
        
        prompt = f"""You are a medical expert analyzing clinical evidence to justify a diagnostic category selection.

**Selected Diagnostic Category:** {category}
**Current Diagnostic Consideration:** {current_node}

**Complete Patient Clinical Information:**
{patient_summary}

**Relevant Medical Knowledge for {category}:**
"""

        # Add flowchart knowledge for this category
        if flowchart_knowledge:
            for knowledge_key, knowledge_content in flowchart_knowledge.items():
                if isinstance(knowledge_content, dict):
                    prompt += f"\n**{knowledge_key}:**\n"
                    for sub_key, sub_content in knowledge_content.items():
                        prompt += f"• {sub_key}: {sub_content}\n"
                elif isinstance(knowledge_content, str):
                    prompt += f"\n**{knowledge_key}:**\n{knowledge_content}\n"
        
        prompt += f"""

**Task:** Analyze the patient's clinical information against the medical knowledge above to justify why {category} is the appropriate diagnostic category to pursue.

**Instructions:**
• Compare patient's symptoms/signs/risk factors with the {category} criteria above
• Identify specific clinical findings that support or contraindicate {category}
• Explain which elements of the patient presentation match the typical {category} profile
• Consider the laboratory/imaging results in the context of {category}
• Provide medical reasoning for why {category} should be explored further

**Analysis:**
Based on the clinical evidence, explain why {category} is the most appropriate diagnostic pathway for this patient."""

        return prompt
    
