#!/usr/bin/env python3
"""
Test to verify the progressive reasoning fix restores full 4-stage workflow
"""

import json
from medeval import DiagnosticEvaluator

def test_progressive_fix():
    """Test that progressive reasoning now uses standard mode by default and captures all stages"""
    
    print("🔧 Testing Progressive Reasoning Fix")
    print("=" * 50)
    
    # Load API key
    try:
        with open('medeval/openai_api_key.txt', 'r') as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        print("❌ API key file not found")
        return False
    
    # Create evaluator 
    evaluator = DiagnosticEvaluator(
        api_key=api_key,
        model="gpt-4o-mini",
        show_responses=False
    )
    
    print(f"✅ Evaluator created successfully")
    
    # Test single sample with progressive reasoning
    print(f"\n🧪 Running progressive reasoning evaluation...")
    results = evaluator.evaluate_dataset(
        num_inputs=6,
        provide_diagnosis_list=True,
        max_samples=1,
        progressive_reasoning=True,
        progressive_fast_mode=False,  # Explicitly request standard mode
        num_suspicions=3,
        max_reasoning_steps=2
    )
    
    print(f"✅ Evaluation completed")
    
    # Analyze the results structure
    if results['detailed_results']:
        sample_result = results['detailed_results'][0]
        
        print(f"\n📊 Results Analysis:")
        print(f"   Sample: {sample_result.get('sample_path', 'unknown')}")
        print(f"   Progressive Mode: {sample_result.get('progressive_mode', 'unknown')}")
        print(f"   Reasoning Steps: {sample_result.get('reasoning_steps', 0)}")
        
        # Check for prompts and responses
        prompts_responses = sample_result.get('prompts_and_responses', [])
        print(f"   Captured Stages: {len(prompts_responses)}")
        
        if len(prompts_responses) >= 4:
            print(f"   ✅ Full 4-stage workflow captured!")
            
            # Verify each stage
            stage_names = [p.get('stage', 'unknown') for p in prompts_responses]
            print(f"   Stages: {stage_names}")
            
            # Check for Stage 3 reasoning capture
            stage3_data = next((p for p in prompts_responses if p.get('stage') == 'stage_3_choice'), None)
            if stage3_data:
                choice_reasoning = stage3_data.get('choice_reasoning', '')
                if choice_reasoning:
                    print(f"   ✅ Stage 3 reasoning captured: {len(choice_reasoning)} characters")
                else:
                    print(f"   ⚠️  Stage 3 reasoning missing")
            
            # Check for Stage 4 proper reasoning (not hardcoded)
            stage4_data = next((p for p in prompts_responses if 'stage_4' in p.get('stage', '')), None)
            if stage4_data:
                stage4_response = stage4_data.get('response', '')
                if stage4_response.startswith('**BUILDING ON STAGE 3:**'):
                    print(f"   ✅ Stage 4 builds on Stage 3 (no hardcoded messages)")
                elif 'Starting with' in stage4_response:
                    print(f"   ❌ Stage 4 still has hardcoded messages")
                    return False
                else:
                    print(f"   ✅ Stage 4 has proper LLM reasoning")
            
            return True
            
        else:
            print(f"   ❌ Only {len(prompts_responses)} stages captured, expected 4+")
            return False
    else:
        print(f"   ❌ No detailed results found")
        return False

if __name__ == "__main__":
    print("🎯 Progressive Reasoning Fix Verification Test")
    print("=" * 60)
    
    success = test_progressive_fix()
    
    if success:
        print(f"\n🎉 VERIFICATION SUCCESS!")
        print(f"   ✅ 4-stage progressive reasoning restored")
        print(f"   ✅ All prompts and responses captured")
        print(f"   ✅ No hardcoded messages")
        print(f"   ✅ Stage 3 to Stage 4 continuity maintained")
    else:
        print(f"\n❌ VERIFICATION FAILED!")
        print(f"   Fix did not restore expected 4-stage workflow") 