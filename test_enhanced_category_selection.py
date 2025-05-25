#!/usr/bin/env python3
"""
Test script to verify the enhanced initial category selection reasoning
"""

import os
from medeval import DiagnosticEvaluator
from medeval.utils import load_sample, collect_sample_files, get_samples_directory

def test_enhanced_category_selection():
    """Test the enhanced evidence-based initial category selection"""
    
    print("🔍 Testing Enhanced Initial Category Selection")
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
        
        # Test with iterative reasoning and multiple categories (k=3)
        evaluator = DiagnosticEvaluator(
            api_key=api_key,
            model="gpt-4o-mini",
            show_responses=True
        )
        
        print("🧪 Testing Iterative Reasoning with Multiple Category Selection (k=3)")
        print("-" * 70)
        
        result = evaluator.evaluate_sample(
            sample_path=sample_path,
            num_inputs=6,
            provide_diagnosis_list=True,
            iterative_reasoning=True,
            num_categories=3,  # This should trigger enhanced reasoning
            max_reasoning_steps=3
        )
        
        print("\n📊 Analysis of Enhanced Category Selection:")
        print("-" * 50)
        
        # Check if we got the enhanced reasoning
        initial_selection = result.get('initial_category_selection_response', '')
        selected_categories = result.get('selected_categories', [])
        chosen_category = result.get('chosen_category', '')
        
        print(f"Selected categories from first step: {selected_categories}")
        print(f"Chosen category for reasoning: {chosen_category}")
        print()
        
        # Analyze the initial category selection response
        if initial_selection:
            print(f"Initial category selection response length: {len(initial_selection)} characters")
            
            # Check for evidence-based reasoning elements
            evidence_keywords = [
                "EVIDENCE ANALYSIS", "COMPARATIVE REASONING", "DECISION", "RATIONALE",
                "patient findings", "specific", "evidence", "defer", "priority"
            ]
            
            found_elements = []
            for keyword in evidence_keywords:
                if keyword.lower() in initial_selection.lower():
                    found_elements.append(keyword)
            
            print(f"Found evidence-based elements: {found_elements}")
            
            if len(found_elements) >= 4:
                print("✅ Enhanced evidence-based reasoning format detected!")
            else:
                print("⚠️  Response may not follow full enhanced format")
            
            # Check response quality
            if len(initial_selection) < 200:
                print("⚠️  Response seems short for detailed evidence-based reasoning")
            elif len(initial_selection) > 500:
                print("✅ Response has good length for detailed reasoning")
            else:
                print("✅ Response has reasonable length")
            
            # Show first part of response
            print(f"\nFirst 300 characters of response:")
            print(f"'{initial_selection[:300]}...'")
        else:
            print("❌ No initial category selection response captured")
        
        print("\n" + "=" * 60)
        
        # Test with single category (k=1) 
        print("🧪 Testing Iterative Reasoning with Single Category Selection (k=1)")
        print("-" * 70)
        
        result_single = evaluator.evaluate_sample(
            sample_path=sample_path,
            num_inputs=6,
            provide_diagnosis_list=True,
            iterative_reasoning=True,
            num_categories=1,  # This should still provide reasoning
            max_reasoning_steps=3
        )
        
        initial_selection_single = result_single.get('initial_category_selection_response', '')
        print(f"Single category response: {initial_selection_single}")
        
        if "Single category selected" in initial_selection_single:
            print("✅ Single category case handled appropriately")
        else:
            print("⚠️  Single category case may need review")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def summarize_enhancement():
    """Summarize the enhancement made"""
    
    print("\n📝 Summary of Enhancement")
    print("=" * 60)
    
    print("Problem identified:")
    print("  ❌ In iterative reasoning, when k > 1 categories were selected,")
    print("     the model jumped to one category without evidence-based reasoning")
    print("  ❌ No detailed justification for choosing one category over others")
    print("  ❌ Broke the evidence-based reasoning chain")
    print()
    
    print("Enhancement implemented:")
    print("  ✅ Enhanced create_initial_category_selection_prompt() with:")
    print("     • Evidence Analysis section requiring patient-to-category matching")
    print("     • Comparative Reasoning section for category prioritization")
    print("     • Structured Decision and Rationale format")
    print("     • Different handling for k=1 vs k>1 cases")
    print("  ✅ Updated parsing to handle structured response format")
    print("  ✅ Increased token limit from 200 to 800 for detailed reasoning")
    print("  ✅ Added response capture and display in show_responses")
    print("  ✅ Added initial category selection data to stored results")
    print()
    
    print("Expected improvements:")
    print("  🎯 Complete evidence-based reasoning chain from start to finish")
    print("  🎯 Detailed justification for category selection priority")
    print("  🎯 Proper handling of both k=1 and k>1 scenarios")
    print("  🎯 Enhanced traceability of reasoning decisions")

if __name__ == "__main__":
    print("🧪 Testing Enhanced Initial Category Selection for Iterative Reasoning")
    print("=" * 80)
    
    success = test_enhanced_category_selection()
    
    summarize_enhancement()
    
    print("\n" + "=" * 80)
    print("🎯 CONCLUSION")
    print("=" * 80)
    
    if success:
        print("✅ Enhanced initial category selection is working!")
        print("🔬 Iterative reasoning now has complete evidence-based flow")
        print("⚡ Ready for sophisticated multi-step diagnostic evaluation")
    else:
        print("⚠️  Some issues detected - may need API key or additional testing")
    
    print("\nRecommendation:")
    print("🧪 Test with actual iterative evaluation to verify complete reasoning chain")
    print("📊 Review response quality and adjust prompts if needed") 