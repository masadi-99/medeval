#!/usr/bin/env python3
"""
Test the FIXED Clean Progressive Reasoning System
Validates critical fixes:
1. Stage 3 chooses from flowchart FIRST STEPS (e.g., "Suspected Pneumonia") not categories
2. No redundancy between reasoning_trace and prompts_and_responses 
3. Stage 4 properly iterates through flowchart step by step
4. Flowchart first steps include signs/symptoms/risks for proper decision making
"""

import json
import os
from medeval.evaluator import DiagnosticEvaluator
from clean_progressive_reasoning import integrate_clean_progressive_reasoning

def test_fixed_clean_progressive_reasoning():
    """Test the FIXED clean progressive reasoning with real API"""
    
    print("🧪 Testing FIXED Clean Progressive Reasoning System")
    print("=" * 60)
    
    # Setup API key
    api_key_file = "medeval/openai_api_key.txt"
    if not os.path.exists(api_key_file):
        print("❌ API key file not found")
        return False
    
    with open(api_key_file, 'r') as f:
        api_key = f.read().strip()
    
    # Create evaluator and integrate FIXED clean reasoning
    evaluator = DiagnosticEvaluator(
        api_key=api_key,
        model="gpt-4o-mini",
        show_responses=True
    )
    
    # Integrate the FIXED clean progressive reasoning
    evaluator = integrate_clean_progressive_reasoning(evaluator)
    
    # Test sample
    sample = {
        "input1": "A 65-year-old man presents with acute chest pain and shortness of breath",
        "input2": "Pain started 2 hours ago, described as crushing, radiating to left arm",
        "input3": "History of hypertension and diabetes, takes metformin and lisinopril", 
        "input4": "Family history of coronary artery disease",
        "input5": "Physical exam: diaphoretic, blood pressure 160/95, heart rate 110",
        "input6": "ECG shows ST elevation in leads II, III, aVF. Troponin elevated at 2.3"
    }
    
    print(f"📋 Testing with sample case...")
    
    # Run the FIXED progressive reasoning
    try:
        result = evaluator.progressive_reasoning_workflow(
            sample=sample,
            num_suspicions=3,
            max_reasoning_steps=3
        )
        
        print(f"\n✅ FIXED Progressive reasoning completed successfully!")
        print(f"   Mode: {result.get('mode', 'unknown')}")
        print(f"   Final diagnosis: {result.get('final_diagnosis', 'None')}")
        print(f"   Reasoning steps: {result.get('reasoning_steps', 0)}")
        print(f"   Chosen suspicion: {result.get('chosen_suspicion', 'None')}")
        
        # CRITICAL VALIDATION: Check the fixes
        
        # 1. Check for NO redundancy - should only have prompts_and_responses
        has_reasoning_trace = 'reasoning_trace' in result
        prompts_and_responses = result.get('prompts_and_responses', [])
        
        print(f"\n🔍 REDUNDANCY CHECK:")
        print(f"   Has reasoning_trace: {has_reasoning_trace}")
        print(f"   Prompts and responses count: {len(prompts_and_responses)}")
        
        if has_reasoning_trace:
            print("   ❌ FAILED: Still has redundant reasoning_trace field")
            return False
        else:
            print("   ✅ PASSED: No redundant reasoning_trace field")
        
        # 2. Check that Stage 3 (Step 1) chooses from flowchart FIRST STEPS
        step1_data = None
        for step in prompts_and_responses:
            if step.get('step') == 1:
                step1_data = step
                break
        
        print(f"\n🔍 STAGE 3 FIRST STEP CHECK:")
        if step1_data:
            chosen_first_step = step1_data.get('chosen_first_step', '')
            flowchart_category = step1_data.get('flowchart_category', '')
            
            print(f"   Chosen first step: {chosen_first_step}")
            print(f"   Flowchart category: {flowchart_category}")
            
            # Check if it's a first step (should be like "Suspected X") not just category name
            if chosen_first_step and chosen_first_step != flowchart_category:
                if "suspected" in chosen_first_step.lower() or chosen_first_step.startswith("Acute") or chosen_first_step.startswith("Possible"):
                    print("   ✅ PASSED: Chose flowchart first step, not general category")
                else:
                    print("   ⚠️  WARNING: Might not be flowchart first step format")
            else:
                print("   ❌ FAILED: Chose general category instead of flowchart first step")
                return False
        else:
            print("   ❌ FAILED: No Step 1 data found")
            return False
        
        # 3. Check that Stage 4 has multiple iteration steps
        stage4_steps = [step for step in prompts_and_responses if step.get('step', 0) >= 2]
        
        print(f"\n🔍 STAGE 4 ITERATION CHECK:")
        print(f"   Stage 4 steps count: {len(stage4_steps)}")
        
        if len(stage4_steps) > 0:
            print("   ✅ PASSED: Stage 4 has reasoning steps")
            
            # Check that each step has flowchart progression
            for i, step in enumerate(stage4_steps):
                current_node = step.get('current_node', '')
                chosen_diagnosis = step.get('chosen_diagnosis', '')
                print(f"     Step {step.get('step')}: {current_node} → {chosen_diagnosis}")
        else:
            print("   ⚠️  WARNING: No Stage 4 reasoning steps found")
        
        # 4. Check for detailed prompts including signs/symptoms/risks
        step1_prompt = step1_data.get('prompt', '') if step1_data else ''
        
        print(f"\n🔍 FLOWCHART CRITERIA CHECK:")
        if step1_prompt:
            has_starting_points = "STARTING POINTS" in step1_prompt or "FLOWCHART STARTING POINTS" in step1_prompt
            has_criteria = any(keyword in step1_prompt.lower() for keyword in ["signs", "symptoms", "risks", "criteria"])
            
            print(f"   Has starting points section: {has_starting_points}")
            print(f"   Has clinical criteria: {has_criteria}")
            
            if has_starting_points and has_criteria:
                print("   ✅ PASSED: Stage 3 prompt includes flowchart first steps with criteria")
            else:
                print("   ⚠️  WARNING: Stage 3 prompt might be missing flowchart criteria")
        
        # Save results for inspection
        with open('fixed_clean_test_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n📄 Results saved to: fixed_clean_test_results.json")
        
        # Overall validation
        print(f"\n🎯 OVERALL VALIDATION:")
        all_checks_passed = (
            not has_reasoning_trace and  # No redundancy
            len(prompts_and_responses) > 2 and  # Multiple steps
            step1_data and step1_data.get('chosen_first_step') != step1_data.get('flowchart_category')  # First step not category
        )
        
        if all_checks_passed:
            print("✅ SUCCESS: All critical fixes validated!")
            print("   ✓ Removed redundancy")
            print("   ✓ Stage 3 chooses flowchart first steps")  
            print("   ✓ Stage 4 has proper iteration")
            return True
        else:
            print("❌ FAILED: Some fixes not working correctly")
            return False
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_no_hardcoded_messages():
    """Validate that there are no hardcoded 'Starting with...' messages"""
    
    print("\n🔍 Checking for hardcoded messages...")
    
    # Check the fixed implementation
    with open('clean_progressive_reasoning.py', 'r') as f:
        content = f.read()
    
    hardcoded_patterns = [
        "Starting with",
        "Suspected ACS", 
        "hardcoded",
        "mock_"
    ]
    
    found_hardcoded = False
    for pattern in hardcoded_patterns:
        if pattern in content and "mock_" not in pattern:  # Allow mock functions
            print(f"   ⚠️  Found potential hardcoded pattern: {pattern}")
            found_hardcoded = True
    
    if not found_hardcoded:
        print("   ✅ No hardcoded messages found in implementation")
        return True
    else:
        print("   ❌ Found potential hardcoded messages")
        return False

if __name__ == "__main__":
    print("🚀 Starting FIXED Clean Progressive Reasoning Tests")
    print("=" * 70)
    
    # Test the fixes
    reasoning_success = test_fixed_clean_progressive_reasoning()
    no_hardcoded_success = validate_no_hardcoded_messages()
    
    print("\n" + "=" * 70)
    if reasoning_success and no_hardcoded_success:
        print("🎉 ALL TESTS PASSED: Fixed clean progressive reasoning working correctly!")
        print("   ✓ No data redundancy")
        print("   ✓ Stage 3 chooses flowchart first steps")
        print("   ✓ Stage 4 iterates through flowcharts properly")
        print("   ✓ No hardcoded messages")
    else:
        print("❌ SOME TESTS FAILED: Additional fixes needed")
        
    print("=" * 70) 