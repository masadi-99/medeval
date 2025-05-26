#!/usr/bin/env python3
"""
Test script to verify Stage 3 reasoning capture and Heart Failure mapping fixes
"""

import json
import os
from medeval import DiagnosticEvaluator

def test_heart_failure_mapping():
    """Test that Heart Failure maps to itself, not ACS"""
    
    print("🧪 Testing Heart Failure Category Mapping")
    print("=" * 50)
    
    evaluator = DiagnosticEvaluator(
        api_key="dummy",
        model="gpt-4o-mini", 
        show_responses=False
    )
    
    # Test the exact issue from the user's results
    result = evaluator.map_suspicion_to_category("Heart Failure")
    
    print(f"✅ Heart Failure → {result}")
    assert result == "Heart Failure", f"Expected 'Heart Failure', got '{result}'"
    
    # Test other similar cases
    test_cases = [
        ("Pneumonia", "Pneumonia"),
        ("Acute Coronary Syndrome", "Acute Coronary Syndrome"), 
        ("Tuberculosis", "Tuberculosis"),  # Both TB and Pneumonia are valid categories
        ("Bacterial Pneumonia", "Pneumonia")  # Should map to broader category
    ]
    
    for suspicion, expected in test_cases:
        result = evaluator.map_suspicion_to_category(suspicion)
        print(f"✅ {suspicion} → {result}")
        assert result == expected, f"Expected '{expected}', got '{result}'"
    
    print("✅ All mapping tests passed!")

def test_suspicion_choice_reasoning_extraction():
    """Test that reasoning is properly extracted from suspicion choice responses"""
    
    print("\n🧪 Testing Suspicion Choice Reasoning Extraction")
    print("=" * 50)
    
    evaluator = DiagnosticEvaluator(
        api_key="dummy",
        model="gpt-4o-mini",
        show_responses=False
    )
    
    suspicions = ["Pneumonia", "Acute Coronary Syndrome", "Heart Failure"]
    
    # Test structured response format
    structured_response = """**CHOSEN SUSPICION:** 1 - Pneumonia

**REASONING:** Based on the clinical findings including fever, productive cough, and chest pain, along with the physical examination showing crackles and dullness to percussion, pneumonia is the most consistent diagnosis. The laboratory results showing elevated white blood cell count and the chest X-ray findings of consolidation further support this diagnosis."""
    
    chosen, reasoning = evaluator.parse_suspicion_choice(structured_response, suspicions)
    
    print(f"✅ Chosen suspicion: {chosen}")
    print(f"✅ Reasoning: {reasoning[:100]}...")
    
    assert chosen == "Pneumonia", f"Expected 'Pneumonia', got '{chosen}'"
    assert "fever" in reasoning and "crackles" in reasoning, f"Reasoning missing key details: {reasoning}"
    
    # Test less structured response
    unstructured_response = """I choose option 3 because the patient shows signs of heart failure including edema and elevated BNP levels consistent with cardiac dysfunction."""
    
    chosen2, reasoning2 = evaluator.parse_suspicion_choice(unstructured_response, suspicions)
    
    print(f"✅ Chosen suspicion: {chosen2}")
    print(f"✅ Reasoning: {reasoning2}")
    
    assert chosen2 == "Heart Failure", f"Expected 'Heart Failure', got '{chosen2}'"
    assert "edema" in reasoning2 or "BNP" in reasoning2, f"Reasoning missing key details: {reasoning2}"
    
    print("✅ All reasoning extraction tests passed!")

def test_stage3_prompt_quality():
    """Test that Stage 3 prompt asks for detailed reasoning"""
    
    print("\n🧪 Testing Stage 3 Prompt Quality") 
    print("=" * 50)
    
    evaluator = DiagnosticEvaluator(
        api_key="dummy",
        model="gpt-4o-mini",
        show_responses=False
    )
    
    history_summary = "Patient presents with chest pain and shortness of breath"
    full_summary = "Complete clinical picture with lab results showing elevated troponin"
    suspicions = ["Pneumonia", "Acute Coronary Syndrome", "Heart Failure"]
    recommended_tests = "ECG, chest X-ray, cardiac enzymes"
    
    prompt = evaluator.create_suspicion_choice_prompt(
        history_summary, full_summary, suspicions, recommended_tests
    )
    
    print("✅ Prompt created successfully")
    
    # Check that prompt asks for reasoning
    assert "reasoning" in prompt.lower(), "Prompt should ask for reasoning"
    assert "format:" in prompt.lower(), "Prompt should specify format"
    assert "chosen suspicion:" in prompt.lower(), "Prompt should ask for chosen suspicion"
    
    print("✅ Stage 3 prompt includes reasoning requirements")

if __name__ == "__main__":
    test_heart_failure_mapping()
    test_suspicion_choice_reasoning_extraction()
    test_stage3_prompt_quality()
    print("\n🎉 All Stage 3 fixes working correctly!") 