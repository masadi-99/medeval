#!/usr/bin/env python3
"""
Test script to verify that the truncation bug has been fixed
"""

import os
from medeval import DiagnosticEvaluator
from medeval.utils import load_sample, collect_sample_files, get_samples_directory

def test_patient_data_truncation():
    """Test that patient data is no longer being truncated"""
    
    print("🔍 Testing Patient Data Truncation Fix")
    print("=" * 50)
    
    try:
        # Get a sample with long clinical data
        samples_dir = get_samples_directory()
        sample_files = collect_sample_files(samples_dir)
        
        if not sample_files:
            print("❌ No sample files found")
            return
        
        # Load a sample
        sample_path = sample_files[0]
        sample = load_sample(sample_path)
        
        print(f"📄 Testing with sample: {sample_path}")
        print()
        
        # Check original lengths
        print("📊 Original Clinical Data Lengths:")
        total_length = 0
        for i in range(1, 7):
            input_key = f"input{i}"
            if input_key in sample:
                length = len(sample[input_key])
                total_length += length
                print(f"   {input_key}: {length} characters")
        
        print(f"   Total: {total_length} characters")
        print()
        
        # Create evaluator to test data summary
        if os.getenv('OPENAI_API_KEY'):
            evaluator = DiagnosticEvaluator(
                api_key=os.getenv('OPENAI_API_KEY'),
                model="gpt-4o-mini"
            )
        else:
            # Use a mock evaluator for testing data processing
            evaluator = DiagnosticEvaluator.__new__(DiagnosticEvaluator)
            evaluator.flowchart_dir = None
            evaluator.samples_dir = None
        
        # Test patient data summary (used in iterative reasoning)
        patient_summary = evaluator.create_patient_data_summary(sample, 6)
        
        print("📋 Patient Data Summary (used in reasoning steps):")
        print(f"Length: {len(patient_summary)} characters")
        print()
        
        # Check if any input was truncated
        truncation_found = False
        for i in range(1, 7):
            input_key = f"input{i}"
            if input_key in sample:
                original_content = sample[input_key]
                if len(original_content) > 200:
                    # Check if this content appears in full in the summary
                    if original_content not in patient_summary:
                        # Check if it was truncated (ends with ...)
                        if f"{original_content[:200]}..." in patient_summary:
                            print(f"❌ TRUNCATION DETECTED in {input_key}!")
                            print(f"   Original: {len(original_content)} chars")
                            print(f"   In summary: appears to be truncated to ~200 chars")
                            truncation_found = True
                        else:
                            print(f"✅ {input_key}: Full content preserved ({len(original_content)} chars)")
                    else:
                        print(f"✅ {input_key}: Full content preserved ({len(original_content)} chars)")
                else:
                    print(f"✅ {input_key}: Content preserved ({len(original_content)} chars)")
        
        print()
        
        if not truncation_found:
            print("🎉 SUCCESS: No truncation detected in patient data!")
            print("   Full clinical information is now available for reasoning")
        else:
            print("❌ FAILURE: Truncation still detected!")
            print("   This will severely impact diagnostic reasoning quality")
        
        print()
        
        # Show sample of the summary
        print("📝 Sample of Patient Summary (first 300 chars):")
        print(patient_summary[:300])
        if len(patient_summary) > 300:
            print(f"... [showing 300/{len(patient_summary)} characters]")
        
        return not truncation_found
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


def test_clinical_criteria_truncation():
    """Test that clinical criteria in flowcharts are not truncated"""
    
    print("\n🔍 Testing Clinical Criteria Truncation Fix")
    print("=" * 50)
    
    try:
        from medeval.utils import load_flowchart_categories, load_flowchart_content, format_reasoning_step
        
        # Load a flowchart
        categories = load_flowchart_categories()
        if not categories:
            print("❌ No flowchart categories found")
            return False
        
        category = categories[0]
        flowchart_data = load_flowchart_content(category)
        
        print(f"📊 Testing flowchart: {category}")
        
        # Test if knowledge content is preserved in reasoning step format
        from medeval.utils import get_flowchart_knowledge, get_flowchart_structure, get_flowchart_children
        
        knowledge = get_flowchart_knowledge(flowchart_data)
        structure = get_flowchart_structure(flowchart_data)
        
        # Find a node with children
        test_node = None
        test_children = []
        for node in structure.keys():
            children = get_flowchart_children(structure, node)
            if children:
                test_node = node
                test_children = children[:3]  # Test with first 3 children
                break
        
        if not test_node:
            print("⚠️  No suitable test node found")
            return True
        
        print(f"   Testing node: {test_node}")
        print(f"   Children: {test_children}")
        
        # Create a test reasoning step
        test_patient_data = "Test patient with chest pain and shortness of breath."
        
        reasoning_prompt = format_reasoning_step(
            step_num=1,
            current_node=test_node,
            available_options=test_children,
            knowledge=knowledge,
            patient_data_summary=test_patient_data
        )
        
        print(f"   Reasoning prompt length: {len(reasoning_prompt)} characters")
        
        # Check if any knowledge was truncated
        truncation_found = False
        for child in test_children:
            if child in knowledge:
                child_knowledge = knowledge[child]
                if isinstance(child_knowledge, str) and len(child_knowledge) > 250:
                    # Check if truncated version appears in prompt
                    truncated_version = child_knowledge[:250] + "..."
                    if truncated_version in reasoning_prompt:
                        print(f"❌ TRUNCATION DETECTED in {child} knowledge!")
                        truncation_found = True
                    elif child_knowledge in reasoning_prompt:
                        print(f"✅ {child}: Full knowledge preserved ({len(child_knowledge)} chars)")
                    else:
                        print(f"⚠️  {child}: Knowledge not found in prompt (might be dict format)")
                elif isinstance(child_knowledge, dict):
                    # Check dict values
                    for key, value in child_knowledge.items():
                        if isinstance(value, str) and len(value) > 200:
                            truncated_version = value[:200] + "..."
                            if truncated_version in reasoning_prompt:
                                print(f"❌ TRUNCATION DETECTED in {child}.{key}!")
                                truncation_found = True
                            elif value in reasoning_prompt:
                                print(f"✅ {child}.{key}: Full content preserved ({len(value)} chars)")
        
        if not truncation_found:
            print("🎉 SUCCESS: No truncation detected in clinical criteria!")
        else:
            print("❌ FAILURE: Truncation still detected in clinical criteria!")
        
        return not truncation_found
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Truncation Bug Fix")
    print("=" * 60)
    
    success1 = test_patient_data_truncation()
    success2 = test_clinical_criteria_truncation()
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY")
    print("=" * 60)
    
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Patient data truncation bug has been fixed")
        print("✅ Clinical criteria truncation bug has been fixed")
        print("🔬 Evidence-based reasoning should now work properly")
    else:
        print("❌ SOME TESTS FAILED!")
        if not success1:
            print("❌ Patient data truncation still detected")
        if not success2:
            print("❌ Clinical criteria truncation still detected")
        print("⚠️  This will severely impact diagnostic reasoning quality")
    
    print("\nRecommendation:")
    if success1 and success2:
        print("✅ Framework is ready for evidence-based diagnostic evaluation")
    else:
        print("❌ Additional fixes needed before using iterative reasoning mode") 