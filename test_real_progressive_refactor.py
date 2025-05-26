#!/usr/bin/env python3
"""
Real API test to verify the fundamental progressive reasoning refactor works
"""

import json
import os
from medeval import DiagnosticEvaluator

def load_api_key():
    """Load API key from file"""
    try:
        with open('openai_api_key.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("⚠️  openai_api_key.txt not found. Please create this file with your OpenAI API key.")
        return None

def test_real_progressive_reasoning():
    """Test progressive reasoning with real API to verify refactor works"""
    
    print("🧪 Testing Real Progressive Reasoning with API")
    print("=" * 60)
    
    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("❌ Cannot run test without API key")
        return False
    
    # Create evaluator with real API key
    evaluator = DiagnosticEvaluator(
        api_key=api_key,
        model="gpt-4o-mini",
        show_responses=True,  # Show responses to see the actual reasoning
        progressive_reasoning=True
    )
    
    print(f"✅ Evaluator created with real API key")
    print(f"📊 Found {len(evaluator.flowchart_categories)} disease categories")
    print(f"📋 Found {len(evaluator.possible_diagnoses)} possible diagnoses")
    
    # Run evaluation with small sample size
    print("\n🔬 Running progressive reasoning evaluation (max 2 samples)...")
    results = evaluator.evaluate_dataset(
        num_inputs=6,
        provide_diagnosis_list=True,
        max_samples=2,  # Small number since model is slow
        progressive_reasoning=True,
        progressive_fast_mode=False,  # Use full progressive reasoning, not fast mode
        num_suspicions=3,
        max_reasoning_steps=3
    )
    
    print("\n📊 Evaluation Results:")
    print(f"   Overall Accuracy: {results['overall_metrics']['accuracy']:.3f}")
    print(f"   Number of samples: {results['overall_metrics']['num_samples']}")
    
    # Analyze detailed results to check for hardcoded messages
    print("\n🔍 Analyzing reasoning traces for hardcoded messages...")
    
    found_hardcoded = False
    found_stage3_continuity = False
    
    for i, result in enumerate(results['detailed_results']):
        print(f"\n📁 Sample {i+1}: {os.path.basename(result['sample_path'])}")
        print(f"   Ground Truth: {result['ground_truth']}")
        print(f"   Predicted: {result['predicted_matched']}")
        print(f"   Correct: {result['correct']}")
        
        if 'reasoning_trace' in result:
            print(f"   Reasoning Steps: {len(result['reasoning_trace'])}")
            
            for step_idx, step in enumerate(result['reasoning_trace']):
                response = step.get('response', '')
                action = step.get('action', '')
                
                print(f"   Step {step_idx + 1}:")
                print(f"     Action: {action}")
                print(f"     Response preview: {response[:100]}...")
                
                # Check for hardcoded "Starting with..." messages
                if response.startswith("Starting with"):
                    print(f"     ❌ FOUND HARDCODED MESSAGE: {response}")
                    found_hardcoded = True
                
                # Check for Stage 3 continuity in Stage 4
                if action in ['stage4_initial_reasoning', 'initial_reasoning']:
                    if 'Stage 3' in response or 'chosen' in response.lower() or 'building on' in response.lower():
                        print(f"     ✅ FOUND STAGE 3 CONTINUITY: References previous reasoning")
                        found_stage3_continuity = True
                    else:
                        print(f"     ⚠️  NO STAGE 3 CONTINUITY DETECTED")
        
        # Check Stage 3 reasoning capture
        if 'prompts_and_responses' in result:
            stage3_data = None
            for stage_data in result['prompts_and_responses']:
                if stage_data.get('stage') == 'stage_3_choice':
                    stage3_data = stage_data
                    break
            
            if stage3_data:
                choice_reasoning = stage3_data.get('choice_reasoning', '')
                if choice_reasoning:
                    print(f"   ✅ STAGE 3 REASONING CAPTURED: {len(choice_reasoning)} characters")
                else:
                    print(f"   ❌ STAGE 3 REASONING MISSING")
            else:
                print(f"   ⚠️  Stage 3 data not found in prompts_and_responses")
    
    # Final assessment
    print("\n🎯 Refactor Assessment:")
    if found_hardcoded:
        print("❌ FAILED: Still found hardcoded 'Starting with...' messages")
        return False
    else:
        print("✅ SUCCESS: No hardcoded messages found")
    
    if found_stage3_continuity:
        print("✅ SUCCESS: Found Stage 3 continuity in Stage 4 reasoning")
    else:
        print("❌ FAILED: No Stage 3 continuity detected")
        return False
    
    return True

def test_category_mapping_real():
    """Test category mapping with real API"""
    
    print("\n🧪 Testing Category Mapping with Real API")
    print("=" * 50)
    
    api_key = load_api_key()
    if not api_key:
        return False
    
    evaluator = DiagnosticEvaluator(
        api_key=api_key,
        model="gpt-4o-mini",
        show_responses=False
    )
    
    # Test the specific mapping issues
    test_cases = [
        ("Heart Failure", "Heart Failure"),
        ("Tuberculosis", "Tuberculosis"),
        ("Pneumonia", "Pneumonia"),
        ("Bacterial Pneumonia", "Pneumonia")  # Should map to broader category
    ]
    
    all_passed = True
    for suspicion, expected in test_cases:
        result = evaluator.map_suspicion_to_category(suspicion)
        if result == expected:
            print(f"✅ {suspicion} → {result}")
        else:
            print(f"❌ {suspicion} → {result} (expected {expected})")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("🚀 Real API Testing of Fundamental Progressive Reasoning Refactor")
    print("=" * 70)
    
    # Test progressive reasoning
    progressive_success = test_real_progressive_reasoning()
    
    # Test category mapping
    mapping_success = test_category_mapping_real()
    
    print(f"\n🎉 Final Results:")
    print(f"   Progressive Reasoning Refactor: {'✅ PASS' if progressive_success else '❌ FAIL'}")
    print(f"   Category Mapping: {'✅ PASS' if mapping_success else '❌ FAIL'}")
    
    if progressive_success and mapping_success:
        print(f"\n🎯 FUNDAMENTAL REFACTOR VERIFICATION: ✅ SUCCESS")
        print(f"   • No hardcoded messages found")
        print(f"   • Stage 3 to Stage 4 continuity confirmed") 
        print(f"   • Category mapping working correctly")
        print(f"   • Real API integration successful")
    else:
        print(f"\n🎯 FUNDAMENTAL REFACTOR VERIFICATION: ❌ ISSUES DETECTED")
        print(f"   Please review the specific failures above") 