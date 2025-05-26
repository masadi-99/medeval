#!/usr/bin/env python3
"""
Test script to verify that tuberculosis suspicions are correctly mapped to pneumonia category
"""

import json
import os
from medeval import DiagnosticEvaluator

def test_tuberculosis_mapping():
    """Test that tuberculosis suspicions map to pneumonia category"""
    
    print("🧪 Testing Tuberculosis -> Pneumonia Category Mapping")
    print("=" * 60)
    
    # Create evaluator
    evaluator = DiagnosticEvaluator(
        api_key="dummy",  # Won't actually call API in this test
        model="gpt-4o-mini",
        show_responses=False
    )
    
    print(f"📋 Available flowchart categories: {len(evaluator.flowchart_categories)}")
    for cat in sorted(evaluator.flowchart_categories):
        print(f"   - {cat}")
    print()
    
    # Test the mapping function
    print("🔬 Testing map_suspicion_to_category function:")
    
    test_cases = [
        "Tuberculosis",
        "tuberculosis", 
        "TB",
        "Bacterial Pneumonia",
        "Pneumonia",
        "Viral Pneumonia"
    ]
    
    for suspicion in test_cases:
        mapped_category = evaluator.map_suspicion_to_category(suspicion)
        print(f"   '{suspicion}' -> '{mapped_category}'")
        
        # Verify tuberculosis maps to Pneumonia category
        if 'tuberculosis' in suspicion.lower() or suspicion.lower() == 'tb':
            if mapped_category and 'pneumonia' in mapped_category.lower():
                print(f"     ✅ Tuberculosis correctly mapped to Pneumonia category")
            else:
                print(f"     ❌ Tuberculosis should map to Pneumonia category")
                return False
    
    print()
    
    # Test the progressive reasoning logic
    print("🔬 Testing progressive_iterative_reasoning logic:")
    
    # Mock sample data
    sample = {
        'input1': 'Cough and fever for 3 weeks',
        'input2': 'Patient presents with persistent cough, night sweats, and weight loss',
        'input3': 'No significant past medical history',
        'input4': 'No family history of tuberculosis',
        'input5': 'Physical exam shows fever, weight loss, lung crackles',
        'input6': 'Chest X-ray shows infiltrates, sputum pending'
    }
    
    # Test that tuberculosis suspicion gets mapped correctly
    chosen_suspicion = "Tuberculosis"
    
    print(f"   Testing chosen suspicion: '{chosen_suspicion}'")
    
    # Check the logic in progressive_iterative_reasoning
    suspected_category = evaluator.map_suspicion_to_category(chosen_suspicion)
    print(f"   First priority mapping result: '{suspected_category}'")
    
    if suspected_category is None and chosen_suspicion in evaluator.flowchart_categories:
        suspected_category = chosen_suspicion
        print(f"   Second priority result: '{suspected_category}'")
    
    # Verify the fix works
    if suspected_category and 'pneumonia' in suspected_category.lower():
        print(f"   ✅ Progressive reasoning will use Pneumonia flowchart")
        print(f"     This should lead to correct diagnosis like 'Bacterial Pneumonia'")
    elif suspected_category == "Tuberculosis":
        print(f"   ❌ Progressive reasoning will use Tuberculosis flowchart")
        print(f"     This would lead to incorrect 'Tuberculosis' diagnosis for pneumonia cases")
        return False
    else:
        print(f"   ⚠️  Unexpected category mapping: '{suspected_category}'")
        return False
    
    return True

def test_flowchart_structure():
    """Test the actual flowchart structures to understand the diagnosis paths"""
    
    print("\n🧪 Testing Flowchart Structures")
    print("=" * 60)
    
    try:
        from medeval.utils import load_flowchart_content
        
        # Load Pneumonia flowchart
        pneumonia_data = load_flowchart_content("Pneumonia")
        print("📊 Pneumonia flowchart structure:")
        print(f"   Diagnostic tree: {pneumonia_data.get('diagnostic', {})}")
        
        # Extract possible diagnoses from Pneumonia flowchart
        def extract_leaf_diagnoses(node, path=""):
            diagnoses = []
            if isinstance(node, dict):
                for key, value in node.items():
                    current_path = f"{path} -> {key}" if path else key
                    if isinstance(value, list) and len(value) == 0:  # leaf node
                        diagnoses.append(current_path)
                    elif isinstance(value, dict):
                        diagnoses.extend(extract_leaf_diagnoses(value, current_path))
            return diagnoses
        
        pneumonia_diagnoses = extract_leaf_diagnoses(pneumonia_data.get('diagnostic', {}))
        print(f"   Possible diagnoses in Pneumonia flowchart:")
        for diag in pneumonia_diagnoses:
            print(f"     - {diag}")
        print()
        
        # Load Tuberculosis flowchart
        tuberculosis_data = load_flowchart_content("Tuberculosis")
        print("📊 Tuberculosis flowchart structure:")
        print(f"   Diagnostic tree: {tuberculosis_data.get('diagnostic', {})}")
        
        tuberculosis_diagnoses = extract_leaf_diagnoses(tuberculosis_data.get('diagnostic', {}))
        print(f"   Possible diagnoses in Tuberculosis flowchart:")
        for diag in tuberculosis_diagnoses:
            print(f"     - {diag}")
        print()
        
        # Verify that Bacterial Pneumonia is in Pneumonia flowchart
        has_bacterial_pneumonia = any('Bacterial Pneumonia' in diag for diag in pneumonia_diagnoses)
        if has_bacterial_pneumonia:
            print("✅ 'Bacterial Pneumonia' found in Pneumonia flowchart")
        else:
            print("❌ 'Bacterial Pneumonia' NOT found in Pneumonia flowchart")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading flowcharts: {e}")
        return False

def main():
    print("🧪 Running Tuberculosis-Pneumonia Mapping Fix Test")
    print("=" * 80)
    
    success1 = test_tuberculosis_mapping()
    success2 = test_flowchart_structure()
    
    overall_success = success1 and success2
    
    print("\n" + "=" * 80)
    if overall_success:
        print("✅ All tests passed!")
        print("🎉 Tuberculosis suspicions will now correctly map to Pneumonia category!")
        print("\n📋 Summary:")
        print("   • Tuberculosis mapping fix: ✅")
        print("   • Flowchart structure verification: ✅")
        print("   • Progressive reasoning should now reach correct diagnoses ✅")
    else:
        print("❌ Some tests failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 