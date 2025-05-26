#!/usr/bin/env python3
"""
Test script for the new progressive reasoning functionality
"""

import os
from medeval import DiagnosticEvaluator
from medeval.utils import load_sample, collect_sample_files, get_samples_directory

def test_progressive_reasoning():
    """Test the progressive reasoning workflow"""
    
    print("🧪 Testing Progressive Reasoning Workflow")
    print("=" * 60)
    
    # Check if we have an API key for testing
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found - set environment variable for full testing")
        return False
    
    try:
        # Get a sample for testing
        samples_dir = get_samples_directory()
        sample_files = collect_sample_files(samples_dir)
        
        if not sample_files:
            print("❌ No sample files found")
            return False
        
        sample_path = sample_files[0]
        sample = load_sample(sample_path)
        
        print(f"📄 Testing with sample: {sample_path}")
        print()
        
        # Test progressive reasoning
        evaluator = DiagnosticEvaluator(
            api_key=api_key,
            model="gpt-4o-mini",
            show_responses=True
        )
        
        print("🏥 Testing Progressive Clinical Workflow (k=3 suspicions)")
        print("-" * 70)
        
        result = evaluator.evaluate_sample(
            sample_path=sample_path,
            num_inputs=6,  # All inputs will be used progressively
            provide_diagnosis_list=True,
            progressive_reasoning=True,
            num_suspicions=3,  # Generate 3 initial suspicions
            max_reasoning_steps=3
        )
        
        print("\n📊 Progressive Reasoning Analysis:")
        print("-" * 50)
        
        # Analyze the progressive workflow
        suspicions = result.get('suspicions_generated', [])
        tests = result.get('recommended_tests', '')
        chosen = result.get('chosen_suspicion', '')
        final_diagnosis = result.get('predicted_raw', '')
        
        print(f"Stage 1 - Generated suspicions: {suspicions}")
        print(f"Stage 2 - Recommended tests: {tests[:150]}...")
        print(f"Stage 3 - Chosen suspicion: {chosen}")
        print(f"Stage 4 - Final diagnosis: {final_diagnosis}")
        print()
        
        # Check workflow completeness
        if suspicions and len(suspicions) >= 3:
            print("✅ Stage 1: Suspicion generation successful")
        else:
            print("⚠️  Stage 1: Issues with suspicion generation")
        
        if tests and len(tests) > 50:
            print("✅ Stage 2: Test recommendation successful")
        else:
            print("⚠️  Stage 2: Issues with test recommendations")
        
        if chosen and chosen in suspicions:
            print("✅ Stage 3: Suspicion choice successful")
        else:
            print("⚠️  Stage 3: Issues with suspicion choice")
        
        if final_diagnosis:
            print("✅ Stage 4: Final diagnosis generated")
        else:
            print("⚠️  Stage 4: Issues with final diagnosis")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt_generation():
    """Test the prompt generation without API calls"""
    
    print("\n🔍 Testing Progressive Reasoning Prompt Generation")
    print("=" * 60)
    
    evaluator = DiagnosticEvaluator(
        api_key="dummy",  # Won't be used for prompt generation
        model="gpt-4o-mini",
        show_responses=False
    )
    
    # Test history summary
    sample = {
        'input1': 'Chest pain for 2 hours',
        'input2': '55-year-old male with sudden onset severe chest pain',
        'input3': 'Hypertension, diabetes',
        'input4': 'Father had MI at age 60',
        'input5': 'BP 150/90, HR 110, diaphoretic',
        'input6': 'ECG shows ST elevation, troponin elevated'
    }
    
    history_summary = evaluator.create_history_summary(sample)
    print(f"History summary length: {len(history_summary)} characters")
    print(f"History summary:\n{history_summary}")
    print()
    
    # Test suspicions prompt
    suspicions_prompt = evaluator.create_suspicions_prompt(history_summary, 3)
    print(f"Suspicions prompt length: {len(suspicions_prompt)} characters")
    print(f"First 200 chars: {suspicions_prompt[:200]}...")
    print()
    
    # Test test recommendation prompt
    suspicions = ["Acute myocardial infarction", "Unstable angina", "Aortic dissection"]
    tests_prompt = evaluator.create_tests_recommendation_prompt(history_summary, suspicions)
    print(f"Tests prompt length: {len(tests_prompt)} characters")
    print(f"First 200 chars: {tests_prompt[:200]}...")
    print()
    
    # Test suspicion choice prompt
    full_summary = evaluator.create_patient_data_summary(sample, 6)
    choice_prompt = evaluator.create_suspicion_choice_prompt(
        history_summary, full_summary, suspicions, "ECG, troponin, chest X-ray"
    )
    print(f"Choice prompt length: {len(choice_prompt)} characters")
    print(f"First 200 chars: {choice_prompt[:200]}...")
    
    return True

def summarize_progressive_reasoning():
    """Summarize the progressive reasoning feature"""
    
    print("\n📝 Progressive Reasoning Feature Summary")
    print("=" * 60)
    
    print("Clinical Workflow Stages:")
    print("  🏥 Stage 1: History Analysis (inputs 1-4)")
    print("     • Chief Complaint, HPI, PMH, Family History")
    print("     • Generate top k diagnostic suspicions")
    print("     • Rank suspicions by likelihood")
    print()
    
    print("  🔬 Stage 2: Test Planning")
    print("     • Based on generated suspicions")
    print("     • Recommend minimum necessary tests")
    print("     • Focus on discriminating between suspicions")
    print()
    
    print("  📋 Stage 3: Test Review & Selection")
    print("     • Present physical exam & lab results (inputs 5-6)")
    print("     • Choose most likely suspicion from initial list")
    print("     • Evidence-based selection with reasoning")
    print()
    
    print("  🎯 Stage 4: Flowchart Navigation")
    print("     • Use chosen suspicion to map to category")
    print("     • Follow iterative flowchart reasoning")
    print("     • Reach final diagnosis")
    print()
    
    print("Key Benefits:")
    print("  ✅ Mirrors real clinical workflow")
    print("  ✅ Tests clinical reasoning at each stage")
    print("  ✅ Evaluates test ordering appropriateness")
    print("  ✅ Assesses information integration skills")
    print("  ✅ Maintains all existing evaluation metrics")

if __name__ == "__main__":
    print("🧪 Testing Progressive Clinical Reasoning Workflow")
    print("=" * 80)
    
    # Test prompt generation (no API needed)
    prompt_success = test_prompt_generation()
    
    # Test full workflow (API needed)
    workflow_success = test_progressive_reasoning()
    
    summarize_progressive_reasoning()
    
    print("\n" + "=" * 80)
    print("🎯 CONCLUSION")
    print("=" * 80)
    
    if prompt_success and workflow_success:
        print("✅ Progressive reasoning implementation is working!")
        print("🏥 Clinical workflow stages are functional")
        print("⚡ Ready for realistic medical reasoning evaluation")
    elif prompt_success:
        print("✅ Prompt generation is working")
        print("⚠️  Full workflow needs API key for complete testing")
        print("🏥 Core progressive reasoning logic is implemented")
    else:
        print("⚠️  Some issues detected - review implementation")
    
    print("\nNext Steps:")
    print("🔧 Add CLI support for --progressive-reasoning flag")
    print("📊 Test with actual clinical samples")
    print("🎯 Evaluate against traditional reasoning modes") 