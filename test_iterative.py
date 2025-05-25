#!/usr/bin/env python3
"""
Simple test script for iterative reasoning functionality
"""

import os
from medeval import DiagnosticEvaluator

def test_iterative_reasoning():
    """Test basic iterative reasoning functionality"""
    
    print("🧪 Testing Iterative Reasoning Functionality")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Skipping API test.")
        return
    
    try:
        # Initialize evaluator
        evaluator = DiagnosticEvaluator(
            api_key=api_key,
            model="gpt-4o-mini",
            use_llm_judge=True,
            show_responses=True  # Show detailed responses for testing
        )
        
        print(f"✅ Evaluator initialized")
        print(f"   Found {len(evaluator.possible_diagnoses)} possible diagnoses")
        print(f"   Found {len(evaluator.flowchart_categories)} flowchart categories")
        
        # Test iterative reasoning on a single sample
        print(f"\n🔬 Testing iterative reasoning on 1 sample...")
        
        results = evaluator.evaluate_dataset(
            num_inputs=6,
            provide_diagnosis_list=True,
            max_samples=1,
            iterative_reasoning=True,
            num_categories=3,
            max_reasoning_steps=4
        )
        
        print(f"\n📊 Results:")
        print(f"   Accuracy: {results['overall_metrics']['accuracy']:.3f}")
        print(f"   Total samples: {results['overall_metrics']['num_samples']}")
        
        if 'category_selection_accuracy' in results['overall_metrics']:
            print(f"   Category selection accuracy: {results['overall_metrics']['category_selection_accuracy']:.3f}")
        
        if 'reasoning_path_accuracy' in results['overall_metrics']:
            print(f"   Reasoning path accuracy: {results['overall_metrics']['reasoning_path_accuracy']:.3f}")
        
        if 'avg_reasoning_steps' in results['overall_metrics']:
            print(f"   Average reasoning steps: {results['overall_metrics']['avg_reasoning_steps']:.1f}")
        
        # Show detailed results for the sample
        if results['detailed_results']:
            sample = results['detailed_results'][0]
            print(f"\n📄 Sample details:")
            print(f"   Ground truth: {sample['ground_truth']}")
            print(f"   Disease category: {sample['disease_category']}")
            print(f"   Selected categories: {sample.get('selected_categories', [])}")
            print(f"   Final diagnosis: {sample['predicted_matched']}")
            print(f"   Correct: {sample['correct']}")
            
            if 'reasoning_trace' in sample:
                print(f"   Reasoning steps: {sample.get('reasoning_steps', 0)}")
                print(f"   Detailed reasoning trace:")
                for step in sample['reasoning_trace']:
                    if step.get('action') == 'start':
                        print(f"     Step {step['step']}: Starting with {step.get('category')} -> {step.get('current_node')}")
                    elif step.get('action') == 'reasoning_step':
                        print(f"     Step {step['step']}: {step.get('current_node')} -> {step.get('chosen_option')}")
                        if step.get('evidence_matching'):
                            print(f"       Evidence Matching: {step['evidence_matching'][:100]}...")
                        if step.get('comparative_analysis'):
                            print(f"       Comparative Analysis: {step['comparative_analysis'][:100]}...")
                        if step.get('rationale'):
                            print(f"       Rationale: {step['rationale'][:100]}...")
                    elif step.get('action') == 'final_diagnosis':
                        print(f"     Step {step['step']}: Final diagnosis -> {step.get('current_node')}")
        
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_iterative_reasoning() 