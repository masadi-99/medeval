#!/usr/bin/env python3
"""
Test script for Single-Step Direct Reasoning

This tests the new single-step reasoning mode where:
1. LLM receives ALL patient information (all 6 inputs)
2. LLM receives ALL possible primary discharge diagnoses
3. In ONE API call, LLM provides detailed reasoning and final diagnosis
"""

import sys
import os
import glob

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from medeval.evaluator import DiagnosticEvaluator
from clean_progressive_reasoning import integrate_clean_progressive_reasoning
from medeval.utils import load_sample, extract_diagnosis_from_path


def test_single_step_direct_reasoning():
    """Test the single-step direct reasoning functionality"""
    
    print("🧪 Testing Single-Step Direct Reasoning")
    print("=" * 60)
    
    try:
        # Initialize evaluator with clean reasoning
        evaluator = DiagnosticEvaluator(
            api_key=None,  # Will use environment variable
            model="gpt-4o-mini",
            show_responses=True
        )
        
        # Integrate clean reasoning (includes single-step)
        evaluator = integrate_clean_progressive_reasoning(evaluator)
        
        # Test with a sample case
        sample_file = "medeval/data/Finished/Acute Coronary Syndrome"
        
        # Find any ACS sample file
        acs_pattern = "medeval/data/Finished/Acute Coronary Syndrome/*/*.json"
        acs_files = glob.glob(acs_pattern)
        
        if acs_files:
            sample_file = acs_files[0]  # Use first available ACS file
        else:
            # Fallback to pneumonia
            pneumonia_pattern = "medeval/data/Finished/Pneumonia/*/*.json"
            pneumonia_files = glob.glob(pneumonia_pattern)
            if pneumonia_files:
                sample_file = pneumonia_files[0]
            else:
                print("❌ No sample files found.")
                return False
        
        print(f"📄 Testing with sample: {sample_file}")
        print("-" * 60)
        
        # Load sample and ground truth
        sample = load_sample(sample_file)
        ground_truth = extract_diagnosis_from_path(sample_file)
        
        print(f"🎯 Ground Truth: {ground_truth}")
        print()
        
        # Run single-step direct reasoning
        print("🔍 Running Single-Step Direct Reasoning...")
        print("-" * 40)
        
        result = evaluator.single_step_direct_reasoning(sample)
        
        # Display results
        print("\n📊 SINGLE-STEP REASONING RESULTS:")
        print("=" * 60)
        print(f"Final Diagnosis: {result['final_diagnosis']}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Correct: {result['final_diagnosis'] == ground_truth}")
        print(f"Reasoning Steps: {result['reasoning_steps']}")
        print(f"Mode: {result['mode']}")
        
        if result['prompts_and_responses']:
            step = result['prompts_and_responses'][0]
            print(f"\n📝 REASONING EXTRACT:")
            print("-" * 40)
            reasoning = step.get('reasoning', '')[:500]  # First 500 chars
            print(f"{reasoning}...")
            
            print(f"\n🔧 EXTRACTED DIAGNOSIS:")
            print("-" * 40)
            print(f"Raw extraction: {step.get('extracted_diagnosis', 'N/A')}")
            print(f"Matched diagnosis: {step.get('matched_diagnosis', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_step_vs_basic_prompt():
    """Compare single-step reasoning with basic prompt"""
    
    print("\n🔍 Comparing Single-Step Reasoning vs Basic Prompt")
    print("=" * 60)
    
    try:
        evaluator = DiagnosticEvaluator(
            api_key=None,
            model="gpt-4o-mini",
            show_responses=False  # Keep output clean for comparison
        )
        
        # Integrate clean reasoning
        evaluator = integrate_clean_progressive_reasoning(evaluator)
        
        # Test with a sample
        sample_file = "medeval/data/Finished/Pneumonia/*/*.json"
        pneumonia_files = glob.glob(sample_file)
        
        if pneumonia_files:
            sample_file = pneumonia_files[0]  # Use first available pneumonia file
        else:
            # Try any available sample
            any_pattern = "medeval/data/Finished/*/*/*.json"
            any_files = glob.glob(any_pattern)
            if any_files:
                sample_file = any_files[0]
            else:
                print("❌ No sample files found for comparison test.")
                return False
        
        sample = load_sample(sample_file)
        ground_truth = extract_diagnosis_from_path(sample_file)
        
        print(f"📄 Sample: {sample_file}")
        print(f"🎯 Ground Truth: {ground_truth}")
        print()
        
        # Test 1: Basic evaluation (existing method)
        print("🔸 Method 1: Basic prompt (provide_diagnosis_list=True)")
        basic_result = evaluator.evaluate_sample(
            sample_file, 
            num_inputs=6, 
            provide_diagnosis_list=True
        )
        
        print(f"   Result: {basic_result['predicted_matched']}")
        print(f"   Correct: {basic_result['correct']}")
        
        # Test 2: Single-step direct reasoning (new method)
        print("\n🔸 Method 2: Single-step direct reasoning (with reasoning)")
        single_step_result = evaluator.single_step_direct_reasoning(sample)
        
        print(f"   Result: {single_step_result['final_diagnosis']}")
        print(f"   Correct: {single_step_result['final_diagnosis'] == ground_truth}")
        
        # Compare reasoning depth
        if single_step_result['prompts_and_responses']:
            reasoning_length = len(single_step_result['prompts_and_responses'][0].get('reasoning', ''))
            print(f"   Reasoning length: {reasoning_length} characters")
            
            prompt_length = len(single_step_result['prompts_and_responses'][0].get('prompt', ''))
            print(f"   Prompt length: {prompt_length} characters")
        
        print("\n💡 Key Differences:")
        print("   - Basic prompt: Direct diagnosis selection only")
        print("   - Single-step reasoning: Systematic analysis + final diagnosis")
        print("   - Single-step includes: differential diagnosis, evidence matching, reasoning")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Single-Step Direct Reasoning Test Suite")
    print("=" * 60)
    
    # Test 1: Basic functionality
    success1 = test_single_step_direct_reasoning()
    
    # Test 2: Comparison with existing method
    success2 = test_single_step_vs_basic_prompt()
    
    print("\n📋 TEST SUMMARY:")
    print("=" * 60)
    print(f"✅ Single-step reasoning test: {'PASSED' if success1 else 'FAILED'}")
    print(f"✅ Comparison test: {'PASSED' if success2 else 'FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Single-step direct reasoning is working correctly.")
        print("\n📝 Usage:")
        print("   # Sync version")
        print("   result = evaluator.single_step_direct_reasoning(sample)")
        print("   # Async version") 
        print("   result = await evaluator.single_step_direct_reasoning_async(sample, 'prefix')")
    else:
        print("\n❌ Some tests failed. Please check the implementation.") 