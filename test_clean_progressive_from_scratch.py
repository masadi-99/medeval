#!/usr/bin/env python3
"""
Test the clean progressive reasoning system from scratch
"""

import sys
import json
from medeval import DiagnosticEvaluator
from clean_progressive_reasoning import integrate_clean_progressive_reasoning

def load_api_key():
    """Load API key from file"""
    possible_paths = [
        'openai_api_key.txt',
        'medeval/openai_api_key.txt',
        './medeval/openai_api_key.txt'
    ]
    
    for path in possible_paths:
        try:
            with open(path, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            continue
    
    print("⚠️  openai_api_key.txt not found")
    return None

def test_clean_progressive_reasoning():
    """Test the clean progressive reasoning system"""
    
    print("🧪 Testing Clean Progressive Reasoning From Scratch")
    print("=" * 60)
    
    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("❌ Cannot run test without API key")
        return False
    
    # Create evaluator
    evaluator = DiagnosticEvaluator(
        api_key=api_key,
        model="gpt-4o-mini",
        show_responses=True
    )
    
    # Integrate clean progressive reasoning
    print("🔧 Integrating clean progressive reasoning system...")
    evaluator = integrate_clean_progressive_reasoning(evaluator)
    
    # Test on 1 sample with progressive reasoning
    print("\n🔍 Testing progressive reasoning workflow...")
    
    results = evaluator.evaluate_dataset(
        num_inputs=6,
        provide_diagnosis_list=True,
        max_samples=1,
        progressive_reasoning=True,
        progressive_fast_mode=False,  # Should be ignored
        num_suspicions=3,
        max_reasoning_steps=3
    )
    
    print(f"\n📊 Results Summary:")
    print(f"Results keys: {list(results.keys())}")
    print(f"Total samples: {len(results.get('results', []))}")
    
    # Debug: Print full results structure
    print(f"\n🔍 Debug - Full Results Structure:")
    for key, value in results.items():
        if key == 'results':
            print(f"  {key}: {len(value) if isinstance(value, list) else type(value)}")
        elif key == 'detailed_results':
            print(f"  {key}: {len(value) if isinstance(value, list) else type(value)}")
        else:
            print(f"  {key}: {value}")
    
    # Check detailed_results instead of results
    sample_results = results.get('detailed_results', results.get('results', []))
    
    if sample_results:
        sample_result = sample_results[0]
        
        print(f"\n🔍 Sample Analysis:")
        print(f"Ground truth: {sample_result.get('ground_truth')}")
        print(f"Predicted: {sample_result.get('predicted_matched')}")
        print(f"Correct: {sample_result.get('correct')}")
        print(f"Mode: {sample_result.get('mode', 'unknown')}")
        
        # Check reasoning trace structure
        reasoning_trace = sample_result.get('reasoning_trace', [])
        print(f"\n📝 Reasoning Trace Analysis:")
        print(f"Total steps: {len(reasoning_trace)}")
        
        all_steps_have_prompts = True
        for i, step in enumerate(reasoning_trace):
            has_prompt = 'prompt' in step
            has_response = 'response' in step
            step_type = step.get('action', 'unknown')
            
            print(f"  Step {i}: {step_type} - Prompt: {has_prompt}, Response: {has_response}")
            
            if not has_prompt or not has_response:
                all_steps_have_prompts = False
        
        # Check for hardcoded messages
        has_hardcoded = False
        for step in reasoning_trace:
            response = step.get('response', '')
            if 'Starting with' in response and '->' in response:
                has_hardcoded = True
                print(f"⚠️  Found hardcoded message: {response[:100]}...")
                break
        
        # Check suspicions structure
        suspicions = sample_result.get('suspicions', [])
        print(f"\n🎯 Suspicions Analysis:")
        print(f"Suspicions type: {type(suspicions)}")
        print(f"Suspicions: {suspicions}")
        
        # Save detailed results for inspection
        with open('clean_test_results.json', 'w') as f:
            json.dump(sample_result, f, indent=2)
        print(f"\n💾 Detailed results saved to: clean_test_results.json")
        
        # Validation summary
        print(f"\n✅ Validation Results:")
        print(f"   All steps have prompts/responses: {all_steps_have_prompts}")
        print(f"   No hardcoded messages: {not has_hardcoded}")
        print(f"   Clean progressive mode: {sample_result.get('mode') == 'clean_step_by_step'}")
        print(f"   Multiple reasoning steps: {len(reasoning_trace) > 1}")
        
        if all_steps_have_prompts and not has_hardcoded and len(reasoning_trace) > 1:
            print("\n🎉 SUCCESS: Clean progressive reasoning working correctly!")
            return True
        else:
            print("\n❌ ISSUES FOUND: Clean progressive reasoning needs fixes")
            return False
    
    else:
        print("❌ No results generated")
        return False

if __name__ == "__main__":
    success = test_clean_progressive_reasoning()
    sys.exit(0 if success else 1) 