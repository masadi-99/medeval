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
    extract_diagnosis_from_path
)


def main():
    """Main CLI entry point for evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate LLM on diagnostic reasoning tasks')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model to use')
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
    
    args = parser.parse_args()
    
    provide_list = args.provide_list and not args.no_list
    
    try:
        evaluator = DiagnosticEvaluator(
            api_key=args.api_key, 
            model=args.model,
            flowchart_dir=args.flowchart_dir,
            samples_dir=args.samples_dir
        )
        
        results = evaluator.evaluate_dataset(
            num_inputs=args.num_inputs,
            provide_diagnosis_list=provide_list,
            max_samples=args.max_samples,
            samples_dir=args.samples_dir
        )
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {results['configuration']['model']}")
        print(f"Number of inputs used: {results['configuration']['num_inputs']}")
        print(f"Provided diagnosis list: {results['configuration']['provide_diagnosis_list']}")
        print(f"Total samples: {results['overall_metrics']['num_samples']}")
        print(f"Accuracy: {results['overall_metrics']['accuracy']:.3f}")
        print(f"Precision: {results['overall_metrics']['precision']:.3f}")
        print(f"Recall: {results['overall_metrics']['recall']:.3f}")
        print(f"F1-Score: {results['overall_metrics']['f1']:.3f}")
        
        evaluator.save_results(results, args.output)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have set up the data correctly and have a valid API key")


def demo_main():
    """Demo CLI entry point"""
    parser = argparse.ArgumentParser(description='Run demo of LLM diagnostic evaluator')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--samples-dir', help='Custom samples directory')
    parser.add_argument('--flowchart-dir', help='Custom flowchart directory')
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Error: Please set your OpenAI API key")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   or use --api-key argument")
        return
    
    try:
        run_demo(api_key, args.samples_dir, args.flowchart_dir)
    except Exception as e:
        print(f"❌ Error running demo: {e}")


def run_demo(api_key: str, samples_dir: str = None, flowchart_dir: str = None):
    """Run a demo evaluation with a small subset of samples"""
    
    print("🏥 LLM Diagnostic Reasoning Evaluator Demo")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = DiagnosticEvaluator(
        api_key=api_key, 
        model="gpt-4o-mini",
        flowchart_dir=flowchart_dir,
        samples_dir=samples_dir
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
    
    # Demo 3: Evaluate without diagnosis list
    print("\n🔬 Demo 3: No diagnosis list (6 inputs, no list)")
    results_3 = evaluator.evaluate_dataset(
        num_inputs=6,
        provide_diagnosis_list=False,
        max_samples=5,
        samples_dir=samples_dir
    )
    
    print(f"   Accuracy: {results_3['overall_metrics']['accuracy']:.3f}")
    print(f"   Precision: {results_3['overall_metrics']['precision']:.3f}")
    print(f"   Recall: {results_3['overall_metrics']['recall']:.3f}")
    
    # Show sample prediction
    if results_1['detailed_results']:
        sample_result = results_1['detailed_results'][0]
        print(f"\n📄 Sample prediction:")
        print(f"   Ground truth: {sample_result['ground_truth']}")
        print(f"   Predicted: {sample_result['predicted_matched']}")
        print(f"   Correct: {sample_result['correct']}")
    
    print("\n✅ Demo completed!")
    print("\nTo run full evaluation, use:")
    print("medeval --api-key YOUR_API_KEY --max-samples 50")


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
            content = sample[input_key][:100] + "..." if len(sample[input_key]) > 100 else sample[input_key]
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