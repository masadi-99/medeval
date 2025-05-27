#!/usr/bin/env python3
"""
Demo: Single-Step Direct Reasoning
Shows the detailed reasoning output from the new single-step mode.
"""

import sys
import os
import glob

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from medeval.evaluator import DiagnosticEvaluator
from clean_progressive_reasoning import integrate_clean_progressive_reasoning
from medeval.utils import load_sample, extract_diagnosis_from_path


def demo_single_step_reasoning():
    """Demonstrate single-step direct reasoning with full output"""
    
    print("🎯 Single-Step Direct Reasoning Demo")
    print("=" * 80)
    
    try:
        # Initialize evaluator
        evaluator = DiagnosticEvaluator(
            api_key=None,  # Uses environment variable
            model="gpt-4o-mini",
            show_responses=False
        )
        
        # Integrate the new reasoning modes
        evaluator = integrate_clean_progressive_reasoning(evaluator)
        
        # Find a sample file
        pattern = "medeval/data/Finished/*/*/*.json"
        files = glob.glob(pattern)
        
        if not files:
            print("❌ No sample files found.")
            return
        
        sample_file = files[0]  # Use first available file
        sample = load_sample(sample_file)
        ground_truth = extract_diagnosis_from_path(sample_file)
        
        print(f"📄 Sample File: {sample_file}")
        print(f"🎯 Ground Truth: {ground_truth}")
        print("=" * 80)
        
        # Run single-step reasoning
        result = evaluator.single_step_direct_reasoning(sample)
        
        print("📊 RESULTS:")
        print("-" * 40)
        print(f"Final Diagnosis: {result['final_diagnosis']}")
        print(f"Correct: {result['final_diagnosis'] == ground_truth}")
        print(f"Mode: {result['mode']}")
        print()
        
        if result['prompts_and_responses']:
            step = result['prompts_and_responses'][0]
            
            print("📝 FULL PROMPT:")
            print("-" * 40)
            prompt = step.get('prompt', '')
            print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
            print()
            
            print("🧠 FULL LLM REASONING:")
            print("-" * 40)
            response = step.get('response', '')
            print(response)
            print()
            
            print("🔧 PARSING DETAILS:")
            print("-" * 40)
            print(f"Extracted Diagnosis: {step.get('extracted_diagnosis', 'N/A')}")
            print(f"Matched Diagnosis: {step.get('matched_diagnosis', 'N/A')}")
            print(f"Response Length: {len(response)} characters")
            print(f"Reasoning Length: {len(step.get('reasoning', ''))} characters")
        
        print("\n" + "=" * 80)
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_single_step_reasoning() 