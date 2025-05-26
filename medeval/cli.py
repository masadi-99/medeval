"""
Command line interface for MedEval package
"""

import os
import argparse
import json
import random
from typing import Dict

from .evaluator import DiagnosticEvaluator
from .utils import (
    load_possible_diagnoses, 
    get_samples_directory,
    collect_sample_files,
    load_sample,
    extract_diagnosis_from_path,
    extract_disease_category_from_path
)


def main():
    """Main CLI entry point for evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate LLM on diagnostic reasoning tasks')
    
    # Model selection arguments
    parser.add_argument('--model', default='gpt-4o-mini', 
                       help='Model name (e.g., gpt-4o-mini, qwen3-30b, Qwen/Qwen3-30B-A3B)')
    parser.add_argument('--provider', choices=['auto', 'openai', 'huggingface', 'huggingface_api'], 
                       default='auto', help='Model provider (auto-detected by default)')
    
    # API keys and authentication
    parser.add_argument('--api-key', help='OpenAI API key (uses OPENAI_API_KEY env var if not provided)')
    parser.add_argument('--huggingface-token', help='HuggingFace token (uses HUGGINGFACE_TOKEN env var if not provided)')
    
    # Model-specific arguments
    parser.add_argument('--device', default='auto', 
                       help='Device for local models (auto, cpu, cuda, etc.)')
    parser.add_argument('--torch-dtype', default='auto', 
                       help='Torch dtype for local models (auto, float16, bfloat16)')
    parser.add_argument('--thinking-mode', action='store_true', default=True,
                       help='Enable thinking mode for compatible models (default: True)')
    parser.add_argument('--no-thinking-mode', action='store_true',
                       help='Disable thinking mode')
    
    # Evaluation arguments
    parser.add_argument('--num-inputs', type=int, default=6, choices=range(1, 7),
                       help='Number of input fields to use (1-6)')
    parser.add_argument('--provide-list', action='store_true', default=True,
                       help='Provide list of possible diagnoses to LLM')
    parser.add_argument('--no-list', action='store_true',
                       help='Do not provide list of possible diagnoses')
    parser.add_argument('--max-samples', type=int, help='Maximum number of samples to evaluate')
    parser.add_argument('--output', default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--samples-dir', help='Custom samples directory')
    parser.add_argument('--flowchart-dir', help='Custom flowchart directory')
    parser.add_argument('--show-responses', action='store_true',
                       help='Show actual LLM responses during evaluation')
    parser.add_argument('--use-llm-judge', action='store_true', default=True,
                       help='Use LLM as judge for evaluation (default: True)')
    parser.add_argument('--no-llm-judge', action='store_true',
                       help='Disable LLM judge and use exact string matching')
    
    # Reasoning mode arguments
    parser.add_argument('--two-step', action='store_true',
                       help='Use two-step reasoning (category selection then diagnosis)')
    parser.add_argument('--iterative', action='store_true',
                       help='Use iterative step-by-step reasoning through flowcharts')
    parser.add_argument('--progressive', action='store_true',
                       help='Use progressive clinical workflow reasoning (history -> suspicions -> tests -> diagnosis)')
    parser.add_argument('--progressive-fast', action='store_true',
                       help='Use fast progressive reasoning (combines stages for better performance)')
    parser.add_argument('--num-categories', type=int, default=3,
                       help='Number of categories to select in two-step/iterative reasoning (default: 3)')
    parser.add_argument('--num-suspicions', type=int, default=3,
                       help='Number of suspicions to generate in progressive reasoning (default: 3)')
    parser.add_argument('--max-reasoning-steps', type=int, default=5,
                       help='Maximum number of reasoning steps for iterative/progressive modes (default: 5)')
    
    # Concurrency arguments
    parser.add_argument('--concurrent', action='store_true',
                       help='Use concurrent processing for faster evaluation (recommended for large datasets)')
    parser.add_argument('--max-concurrent', type=int, default=10,
                       help='Maximum number of concurrent API calls (default: 10)')
    
    args = parser.parse_args()
    
    # Handle thinking mode
    thinking_mode = args.thinking_mode and not args.no_thinking_mode
    
    # Get API keys from arguments or environment variables
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    huggingface_token = args.huggingface_token or os.getenv('HUGGINGFACE_TOKEN')
    
    provide_list = args.provide_list and not args.no_list
    use_llm_judge = args.use_llm_judge and not args.no_llm_judge
    
    # Validate reasoning mode arguments
    reasoning_modes = [args.two_step, args.iterative, args.progressive, args.progressive_fast]
    if sum(reasoning_modes) > 1:
        print("❌ Error: Only one reasoning mode can be specified at a time")
        print("   Choose from: --two-step, --iterative, --progressive, or --progressive-fast")
        return 1
    
    # Set reasoning mode flags
    two_step_reasoning = args.two_step
    iterative_reasoning = args.iterative
    progressive_reasoning = args.progressive or args.progressive_fast
    progressive_fast_mode = args.progressive_fast
    
    # Validation for different reasoning modes
    if (iterative_reasoning or progressive_reasoning) and args.max_reasoning_steps < 1:
        print("❌ Error: --max-reasoning-steps must be at least 1 for iterative/progressive reasoning")
        return 1
    
    if two_step_reasoning and not provide_list:
        print("❌ Error: Two-step reasoning requires --provide-list")
        return 1
    
    if iterative_reasoning and not provide_list:
        print("❌ Error: Iterative reasoning requires --provide-list")
        return 1
    
    if progressive_reasoning and not provide_list:
        print("❌ Error: Progressive reasoning requires --provide-list")
        return 1
    
    # Validation for different reasoning modes
    if args.two_step and args.iterative:
        print("❌ Error: Cannot use both --two-step and --iterative modes simultaneously")
        return
    
    # Show available predefined models
    if args.model == "list":
        from .models import PREDEFINED_MODELS
        print("Available predefined models:")
        for name, config in PREDEFINED_MODELS.items():
            print(f"  {name}: {config.get('model_name', config.get('model', name))} ({config['provider']})")
        return
    
    # Create evaluator
    try:
        evaluator = DiagnosticEvaluator(
            api_key=api_key,
            model=args.model,
            flowchart_dir=args.flowchart_dir,
            samples_dir=args.samples_dir,
            use_llm_judge=use_llm_judge,
            show_responses=args.show_responses,
            max_concurrent=args.max_concurrent,
            provider=args.provider,
            huggingface_token=huggingface_token,
            device=args.device,
            torch_dtype=args.torch_dtype,
            thinking_mode=thinking_mode
        )
        
        if args.concurrent:
            # Use concurrent evaluation
            import asyncio
            results = asyncio.run(evaluator.evaluate_dataset_concurrent(
                num_inputs=args.num_inputs,
                provide_diagnosis_list=provide_list,
                max_samples=args.max_samples,
                samples_dir=args.samples_dir,
                two_step_reasoning=two_step_reasoning,
                num_categories=args.num_categories,
                iterative_reasoning=iterative_reasoning,
                max_reasoning_steps=args.max_reasoning_steps,
                progressive_reasoning=progressive_reasoning,
                num_suspicions=args.num_suspicions,
                progressive_fast_mode=progressive_fast_mode
            ))
        else:
            # Use synchronous evaluation
            results = evaluator.evaluate_dataset(
                num_inputs=args.num_inputs,
                provide_diagnosis_list=provide_list,
                max_samples=args.max_samples,
                samples_dir=args.samples_dir,
                two_step_reasoning=two_step_reasoning,
                num_categories=args.num_categories,
                iterative_reasoning=iterative_reasoning,
                max_reasoning_steps=args.max_reasoning_steps,
                progressive_reasoning=progressive_reasoning,
                num_suspicions=args.num_suspicions,
                progressive_fast_mode=progressive_fast_mode
            )
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {results['configuration']['model']}")
        print(f"Number of inputs used: {results['configuration']['num_inputs']}")
        print(f"Provided diagnosis list: {results['configuration']['provide_diagnosis_list']}")
        print(f"Two-step reasoning: {results['configuration'].get('two_step_reasoning', False)}")
        if results['configuration'].get('two_step_reasoning'):
            print(f"Number of categories selected: {results['configuration'].get('num_categories', 3)}")
        print(f"Iterative reasoning: {results['configuration'].get('iterative_reasoning', False)}")
        if results['configuration'].get('iterative_reasoning'):
            print(f"Max reasoning steps: {results['configuration'].get('max_reasoning_steps', 5)}")
        print(f"Progressive reasoning: {results['configuration'].get('progressive_reasoning', False)}")
        if results['configuration'].get('progressive_reasoning'):
            print(f"Number of suspicions: {results['configuration'].get('num_suspicions', 3)}")
            print(f"Max reasoning steps: {results['configuration'].get('max_reasoning_steps', 5)}")
            if progressive_fast_mode:
                print(f"Fast mode: Enabled (combined stages for better performance)")
            else:
                print(f"Fast mode: Disabled (full 4-stage workflow)")
            
            # Display test overlap metrics
            if 'test_overlap_precision' in results['overall_metrics']:
                print(f"\n📊 Test Overlap Metrics:")
                print(f"  Precision: {results['overall_metrics']['test_overlap_precision']:.3f} (avoiding unnecessary tests)")
                print(f"  Recall: {results['overall_metrics']['test_overlap_recall']:.3f} (not missing necessary tests)")
                print(f"  F1-Score: {results['overall_metrics']['test_overlap_f1']:.3f}")
                print(f"  Jaccard Index: {results['overall_metrics']['test_overlap_jaccard']:.3f}")
                print(f"  Avg Tests Recommended: {results['overall_metrics']['avg_tests_recommended']:.1f}")
                print(f"  Avg Tests Actually Done: {results['overall_metrics']['avg_tests_actual']:.1f}")
                print(f"  Avg Test Overlap: {results['overall_metrics']['avg_tests_overlap']:.1f}")
                print(f"  Avg Unnecessary Tests: {results['overall_metrics']['avg_unnecessary_tests']:.1f}")
                print(f"  Avg Missed Tests: {results['overall_metrics']['avg_missed_tests']:.1f}")
        print(f"Concurrent processing: {args.concurrent}")
        if args.concurrent:
            print(f"Max concurrent requests: {results['configuration'].get('max_concurrent', 10)}")
        print(f"LLM Judge enabled: {results['configuration'].get('use_llm_judge', False)}")
        print(f"Total samples: {results['overall_metrics']['num_samples']}")
        if results['overall_metrics'].get('failed_samples', 0) > 0:
            print(f"Failed samples: {results['overall_metrics']['failed_samples']}")
        print(f"Accuracy: {results['overall_metrics']['accuracy']:.3f}")
        print(f"Precision: {results['overall_metrics']['precision']:.3f}")
        print(f"Recall: {results['overall_metrics']['recall']:.3f}")
        print(f"F1-Score: {results['overall_metrics']['f1']:.3f}")
        
        # Display category selection accuracy for two-step mode
        if results['configuration'].get('two_step_reasoning') and 'category_selection_accuracy' in results['overall_metrics']:
            print(f"Category selection accuracy: {results['overall_metrics']['category_selection_accuracy']:.3f}")
        
        # Display reasoning path accuracy for iterative mode
        if results['configuration'].get('iterative_reasoning') and 'reasoning_path_accuracy' in results['overall_metrics']:
            print(f"Reasoning path accuracy: {results['overall_metrics']['reasoning_path_accuracy']:.3f}")
            print(f"Average reasoning steps: {results['overall_metrics'].get('avg_reasoning_steps', 0):.1f}")
        
        # Display disease category metrics
        if 'category_metrics' in results and results['category_metrics']:
            print("\n" + "="*50)
            print("DISEASE CATEGORY METRICS")
            print("="*50)
            
            # Sort categories by accuracy (descending)
            category_items = sorted(
                results['category_metrics'].items(),
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )
            
            print(f"{'Category':<25} {'Samples':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<8} {'F1':<8}")
            print("-" * 75)
            
            for category, metrics in category_items:
                print(f"{category:<25} {metrics['num_samples']:<8} "
                      f"{metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
                      f"{metrics['recall']:<8.3f} {metrics['f1']:<8.3f}")
            
            # Summary statistics
            avg_accuracy = sum(m['accuracy'] for m in results['category_metrics'].values()) / len(results['category_metrics'])
            best_category = max(results['category_metrics'].items(), key=lambda x: x[1]['accuracy'])
            worst_category = min(results['category_metrics'].items(), key=lambda x: x[1]['accuracy'])
            
            print("-" * 75)
            print(f"Average category accuracy: {avg_accuracy:.3f}")
            print(f"Best performing category: {best_category[0]} ({best_category[1]['accuracy']:.3f})")
            print(f"Worst performing category: {worst_category[0]} ({worst_category[1]['accuracy']:.3f})")
        
        evaluator.save_results(results, args.output)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have set up the data correctly and have a valid API key")


def demo_main():
    """Demo CLI entry point"""
    parser = argparse.ArgumentParser(description='Run demo of LLM diagnostic evaluator')
    parser.add_argument('--model', default='gpt-4o-mini', 
                       help='Model name (e.g., gpt-4o-mini, qwen3-30b)')
    parser.add_argument('--provider', choices=['auto', 'openai', 'huggingface', 'huggingface_api'], 
                       default='auto', help='Model provider (auto-detected by default)')
    parser.add_argument('--api-key', help='OpenAI API key (uses OPENAI_API_KEY env var if not provided)')
    parser.add_argument('--huggingface-token', help='HuggingFace token (uses HUGGINGFACE_TOKEN env var if not provided)')
    parser.add_argument('--device', default='auto', help='Device for local models')
    parser.add_argument('--thinking-mode', action='store_true', default=True,
                       help='Enable thinking mode for compatible models')
    parser.add_argument('--no-thinking-mode', action='store_true',
                       help='Disable thinking mode')
    parser.add_argument('--samples-dir', help='Custom samples directory')
    parser.add_argument('--flowchart-dir', help='Custom flowchart directory')
    parser.add_argument('--show-responses', action='store_true',
                       help='Show actual LLM responses during demo')
    
    args = parser.parse_args()
    
    # Handle thinking mode
    thinking_mode = args.thinking_mode and not args.no_thinking_mode
    
    # Get API keys from arguments or environment variables
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    huggingface_token = args.huggingface_token or os.getenv('HUGGINGFACE_TOKEN')
    
    try:
        run_demo(
            api_key=api_key, 
            model=args.model,
            provider=args.provider,
            huggingface_token=huggingface_token,
            device=args.device,
            thinking_mode=thinking_mode,
            samples_dir=args.samples_dir, 
            flowchart_dir=args.flowchart_dir, 
            show_responses=args.show_responses
        )
    except Exception as e:
        print(f"❌ Error running demo: {e}")


def run_demo(api_key: str = None, model: str = "gpt-4o-mini", provider: str = "auto",
             huggingface_token: str = None, device: str = "auto", thinking_mode: bool = True,
             samples_dir: str = None, flowchart_dir: str = None, show_responses: bool = False):
    """Run a demo evaluation with a small subset of samples"""
    
    print("🏥 LLM Diagnostic Reasoning Evaluator Demo")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = DiagnosticEvaluator(
        api_key=api_key, 
        model=model,
        flowchart_dir=flowchart_dir,
        samples_dir=samples_dir,
        use_llm_judge=True,
        show_responses=show_responses,
        provider=provider,
        huggingface_token=huggingface_token,
        device=device,
        thinking_mode=thinking_mode
    )
    
    print(f"📋 Found {len(evaluator.possible_diagnoses)} possible diagnoses")
    print(f"📁 Sample diagnosis list: {evaluator.possible_diagnoses[:5]}...")
    
    # Demo 1: Evaluate with all inputs and diagnosis list provided
    print("\n🔬 Demo 1: Full information (6 inputs + diagnosis list)")
    results_1 = evaluator.evaluate_dataset(
        num_inputs=6,
        provide_diagnosis_list=True,
        max_samples=5,  # Small demo
        samples_dir=samples_dir
    )
    
    print(f"   Accuracy: {results_1['overall_metrics']['accuracy']:.3f}")
    print(f"   Precision: {results_1['overall_metrics']['precision']:.3f}")
    print(f"   Recall: {results_1['overall_metrics']['recall']:.3f}")
    
    # Demo 2: Evaluate with limited inputs
    print("\n🔬 Demo 2: Limited information (3 inputs + diagnosis list)")
    results_2 = evaluator.evaluate_dataset(
        num_inputs=3,
        provide_diagnosis_list=True,
        max_samples=5,
        samples_dir=samples_dir
    )
    
    print(f"   Accuracy: {results_2['overall_metrics']['accuracy']:.3f}")
    print(f"   Precision: {results_2['overall_metrics']['precision']:.3f}")
    print(f"   Recall: {results_2['overall_metrics']['recall']:.3f}")
    
    # Demo 3: Evaluate without diagnosis list (with LLM judge)
    print("\n🔬 Demo 3: No diagnosis list (6 inputs, LLM judge enabled)")
    results_3 = evaluator.evaluate_dataset(
        num_inputs=6,
        provide_diagnosis_list=False,
        max_samples=5,
        samples_dir=samples_dir
    )
    
    print(f"   Accuracy: {results_3['overall_metrics']['accuracy']:.3f}")
    print(f"   Precision: {results_3['overall_metrics']['precision']:.3f}")
    print(f"   Recall: {results_3['overall_metrics']['recall']:.3f}")
    
    # Demo 4: Two-step reasoning
    print("\n🔬 Demo 4: Two-step reasoning (category selection + diagnosis)")
    results_4 = evaluator.evaluate_dataset(
        num_inputs=6,
        provide_diagnosis_list=True,
        max_samples=5,
        samples_dir=samples_dir,
        two_step_reasoning=True,
        num_categories=3
    )
    
    print(f"   Accuracy: {results_4['overall_metrics']['accuracy']:.3f}")
    print(f"   Precision: {results_4['overall_metrics']['precision']:.3f}")
    print(f"   Recall: {results_4['overall_metrics']['recall']:.3f}")
    if 'category_selection_accuracy' in results_4['overall_metrics']:
        print(f"   Category Selection Accuracy: {results_4['overall_metrics']['category_selection_accuracy']:.3f}")
    
    # Demo 5: Iterative step-by-step reasoning
    print("\n🔬 Demo 5: Iterative step-by-step reasoning (following flowcharts)")
    results_5 = evaluator.evaluate_dataset(
        num_inputs=6,
        provide_diagnosis_list=True,
        max_samples=3,  # Fewer samples due to complexity
        samples_dir=samples_dir,
        iterative_reasoning=True,
        num_categories=3,
        max_reasoning_steps=4
    )
    
    print(f"   Accuracy: {results_5['overall_metrics']['accuracy']:.3f}")
    print(f"   Precision: {results_5['overall_metrics']['precision']:.3f}")
    print(f"   Recall: {results_5['overall_metrics']['recall']:.3f}")
    if 'category_selection_accuracy' in results_5['overall_metrics']:
        print(f"   Category Selection Accuracy: {results_5['overall_metrics']['category_selection_accuracy']:.3f}")
    if 'reasoning_path_accuracy' in results_5['overall_metrics']:
        print(f"   Reasoning Path Accuracy: {results_5['overall_metrics']['reasoning_path_accuracy']:.3f}")
    if 'avg_reasoning_steps' in results_5['overall_metrics']:
        print(f"   Average Reasoning Steps: {results_5['overall_metrics']['avg_reasoning_steps']:.1f}")
    
    # Show sample prediction
    if results_1['detailed_results']:
        sample_result = results_1['detailed_results'][0]
        print(f"\n📄 Sample prediction:")
        print(f"   Ground truth: {sample_result['ground_truth']}")
        print(f"   Predicted: {sample_result['predicted_matched']}")
        print(f"   Correct: {sample_result['correct']}")
        if show_responses and 'predicted_raw' in sample_result:
            print(f"   Raw response: {sample_result['predicted_raw']}")
    
    # Show two-step sample if available
    if results_4['detailed_results']:
        sample_result = results_4['detailed_results'][0]
        print(f"\n📄 Two-step sample prediction:")
        print(f"   Ground truth category: {sample_result['disease_category']}")
        if 'selected_categories' in sample_result:
            print(f"   Selected categories: {sample_result['selected_categories']}")
            print(f"   Category selection correct: {sample_result['category_selection_correct']}")
        print(f"   Final diagnosis: {sample_result['predicted_matched']}")
        print(f"   Final diagnosis correct: {sample_result['correct']}")
    
    # Show iterative reasoning sample if available
    if results_5['detailed_results']:
        sample_result = results_5['detailed_results'][0]
        print(f"\n📄 Iterative reasoning sample:")
        print(f"   Ground truth category: {sample_result['disease_category']}")
        if 'selected_categories' in sample_result:
            print(f"   Selected categories: {sample_result['selected_categories']}")
        if 'reasoning_trace' in sample_result and sample_result['reasoning_trace']:
            print(f"   Reasoning steps: {sample_result['reasoning_steps']}")
            print(f"   Detailed reasoning path:")
            for step in sample_result['reasoning_trace']:
                if step.get('action') == 'start':
                    print(f"     Step {step['step']}: Starting -> {step.get('current_node')}")
                elif step.get('action') == 'reasoning_step':
                    print(f"     Step {step['step']}: {step.get('current_node')} -> {step.get('chosen_option')}")
                    if step.get('rationale'):
                        print(f"       Rationale: {step['rationale'][:80]}...")
                    if step.get('evidence_matching'):
                        print(f"       Evidence: {step['evidence_matching'][:80]}...")
                elif step.get('action') == 'final_diagnosis':
                    print(f"     Step {step['step']}: Final -> {step.get('current_node')}")
        print(f"   Final diagnosis: {sample_result['predicted_matched']}")
        print(f"   Diagnosis correct: {sample_result['correct']}")
    
    print("\n✅ Demo completed!")
    print("\nTo run full evaluation, use:")
    print("medeval --max-samples 50")
    print("medeval --two-step --max-samples 50  # For two-step reasoning")
    print("medeval --iterative --max-samples 20  # For iterative reasoning")


def show_main():
    """Show sample data CLI entry point"""
    parser = argparse.ArgumentParser(description='Show sample data and analysis')
    parser.add_argument('--samples-dir', help='Custom samples directory')
    parser.add_argument('--flowchart-dir', help='Custom flowchart directory')
    
    args = parser.parse_args()
    
    try:
        show_sample_analysis(args.samples_dir, args.flowchart_dir)
    except Exception as e:
        print(f"❌ Error: {e}")


def show_sample_analysis(samples_dir: str = None, flowchart_dir: str = None):
    """Display sample data analysis"""
    
    print("🏥 MIMIC-IV-Ext-DiReCT Dataset Sample Analysis")
    print("=" * 60)
    
    # Get directories
    try:
        samples_directory = get_samples_directory(samples_dir)
        sample_files = collect_sample_files(samples_directory)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Make sure you have the data files available")
        return
    
    if not sample_files:
        print("❌ No sample files found")
        return
    
    # Pick a random sample
    sample_path = random.choice(sample_files)
    
    print(f"📁 Sample file: {sample_path}")
    print(f"📊 Total samples available: {len(sample_files)}")
    
    # Load and display sample
    sample = load_sample(sample_path)
    
    print(f"\n📋 Clinical Data Structure:")
    print("-" * 30)
    
    input_descriptions = {
        1: "Chief Complaint",
        2: "History of Present Illness", 
        3: "Past Medical History",
        4: "Family History",
        5: "Physical Examination",
        6: "Laboratory Results"
    }
    
    for i in range(1, 7):
        input_key = f"input{i}"
        if input_key in sample:
            # Show FULL clinical content - no truncation for analysis
            content = sample[input_key]
            # Only truncate for display if extremely long (>500 chars)
            if len(content) > 500:
                content = content[:500] + f"... [truncated, full length: {len(sample[input_key])} chars]"
            print(f"   {i}. {input_descriptions[i]}: {repr(content)}")
    
    # Extract ground truth from path
    ground_truth = extract_diagnosis_from_path(sample_path)
    print(f"\n🎯 Ground Truth Diagnosis: {ground_truth}")
    
    # Show possible diagnoses
    try:
        possible_diagnoses = load_possible_diagnoses(flowchart_dir)
        print(f"\n📋 Found {len(possible_diagnoses)} possible diagnoses")
        print("Sample diagnoses:", possible_diagnoses[:10])
        if len(possible_diagnoses) > 10:
            print(f"... and {len(possible_diagnoses) - 10} more")
    except Exception as e:
        print(f"❌ Error loading diagnoses: {e}")
    
    # Show diagnosis distribution
    print(f"\n📊 Diagnosis Distribution:")
    print("-" * 30)
    
    diagnosis_counts = {}
    for sample_file in sample_files:
        diagnosis = extract_diagnosis_from_path(sample_file)
        if diagnosis:
            diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
    
    # Sort by count
    sorted_diagnoses = sorted(diagnosis_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top 10 most common diagnoses:")
    for i, (diagnosis, count) in enumerate(sorted_diagnoses[:10]):
        print(f"{i+1:2d}. {diagnosis:<30} ({count:2d} samples)")
    
    print(f"\n✅ Analysis complete!")
    print(f"\nTo run evaluation:")
    print(f"   medeval-demo                      # Quick demo")
    print(f"   medeval --help                    # Full options")


if __name__ == "__main__":
    main() 