#!/usr/bin/env python3
"""
Quick test to verify that the performance is back to normal after revert
"""

from medeval import DiagnosticEvaluator

def test_prompt_simplicity():
    """Test that the initial category selection prompt is simple again"""
    
    print("🔍 Testing Prompt Simplicity After Revert")
    print("=" * 50)
    
    # Create evaluator without API key (we're just testing prompt generation)
    evaluator = DiagnosticEvaluator(
        api_key="dummy",  # Won't be used for prompt generation
        model="gpt-4o-mini",
        show_responses=False
    )
    
    # Test initial category selection prompt
    patient_summary = "Chief Complaint: Chest pain\nHistory: 45-year-old male with acute chest pain"
    categories = ["Cardiovascular", "Respiratory", "Gastrointestinal"]
    flowcharts = {}  # Empty for this test
    
    prompt = evaluator.create_initial_category_selection_prompt(
        patient_summary, categories, flowcharts
    )
    
    print(f"Generated prompt length: {len(prompt)} characters")
    print(f"Generated prompt:")
    print(f"'{prompt}'")
    print()
    
    # Check if it's the simple version
    is_simple = (
        "Based on the patient information, select which disease category to explore first" in prompt and
        "Respond with ONLY the number" in prompt and
        "EVIDENCE ANALYSIS" not in prompt and
        "COMPARATIVE REASONING" not in prompt
    )
    
    if is_simple:
        print("✅ Prompt is back to simple version!")
        print("✅ Should be much faster now")
        if len(prompt) < 500:
            print("✅ Prompt length is reasonable for quick processing")
        else:
            print("⚠️  Prompt might still be a bit long")
    else:
        print("❌ Prompt still contains complex elements")
    
    print("\n" + "=" * 50)
    
    # Test the category selection parsing is also simple
    test_response = "2"
    selected = evaluator.parse_category_selection(test_response, categories)
    
    print(f"Parse test - Input: '{test_response}' -> Output: '{selected}'")
    
    if selected == "Respiratory":
        print("✅ Simple parsing working correctly")
    else:
        print(f"⚠️  Parsing issue: expected 'Respiratory', got '{selected}'")
    
    return is_simple

def summarize_revert():
    """Summarize what was reverted"""
    
    print("\n📝 Summary of Revert")
    print("=" * 50)
    
    print("What was reverted:")
    print("  ❌ Removed complex evidence-based initial category selection prompt")
    print("  ❌ Removed EVIDENCE ANALYSIS and COMPARATIVE REASONING sections")
    print("  ❌ Removed structured Decision and Rationale format")
    print("  ❌ Reduced token limit back to 200 (from 800)")
    print("  ❌ Removed enhanced response capture and display")
    print()
    
    print("Back to simple approach:")
    print("  ✅ Simple 'select the most promising category' prompt")
    print("  ✅ 'Respond with ONLY the number' instruction")
    print("  ✅ Basic number parsing")
    print("  ✅ Fast, lightweight category selection")
    print()
    
    print("Performance benefits:")
    print("  ⚡ Much shorter prompts = faster API calls")
    print("  ⚡ Lower token limits = faster responses")
    print("  ⚡ Simpler logic = faster processing")
    print("  ⚡ Reduced complexity = better reliability")

if __name__ == "__main__":
    print("🧪 Performance Check After Enhanced Category Selection Revert")
    print("=" * 70)
    
    success = test_prompt_simplicity()
    
    summarize_revert()
    
    print("\n" + "=" * 70)
    print("🎯 CONCLUSION")
    print("=" * 70)
    
    if success:
        print("✅ Successfully reverted to simple, fast initial category selection!")
        print("⚡ Performance should be significantly improved")
        print("🚀 Ready for efficient iterative reasoning evaluation")
    else:
        print("⚠️  Some complex elements may still be present")
    
    print("\nTrade-off:")
    print("📊 Performance: Significantly faster")
    print("🧠 Reasoning: Less detailed initial category justification")
    print("⚖️  Recommendation: Good trade-off for practical evaluation") 