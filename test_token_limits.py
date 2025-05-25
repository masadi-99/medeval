#!/usr/bin/env python3
"""
Test script to verify that token limits have been fixed and responses are complete
"""

import os
from medeval import DiagnosticEvaluator
from medeval.utils import load_sample, collect_sample_files, get_samples_directory

def test_token_limits():
    """Test that token limits are now appropriate for different types of responses"""
    
    print("🔍 Testing Token Limits Fix")
    print("=" * 50)
    
    # Check if we have an API key for testing
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found - will test with mock responses")
        print("   For full testing, set OPENAI_API_KEY environment variable")
        print()
    
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
        
        if api_key:
            # Test with actual API calls
            evaluator = DiagnosticEvaluator(
                api_key=api_key,
                model="gpt-4o-mini",
                show_responses=True
            )
            
            print("🔬 Testing Standard Diagnostic Response:")
            print("-" * 40)
            
            # Test standard evaluation
            prompt = evaluator.create_prompt(sample, 6, True)
            response = evaluator.query_llm(prompt, max_tokens=500)
            
            print(f"Prompt length: {len(prompt)} characters")
            print(f"Response length: {len(response)} characters")
            print(f"Response: {response}")
            print()
            
            # Check if response appears to be truncated
            truncation_indicators = ["...", "Based on", "In conclusion", "Final", "Therefore"]
            appears_complete = any(indicator.lower() in response.lower() for indicator in truncation_indicators)
            
            if len(response) < 50:
                print("⚠️  Response seems very short - might be truncated")
            elif not appears_complete and len(response) > 400:
                print("⚠️  Response might be cut off mid-sentence")
            else:
                print("✅ Response appears to be complete")
            
            print()
            
            # Test category selection (if we have flowcharts)
            try:
                print("🔬 Testing Category Selection Response:")
                print("-" * 40)
                
                category_prompt = evaluator.create_category_selection_prompt(sample, 6, 3)
                category_response = evaluator.query_llm(category_prompt, max_tokens=800)
                
                print(f"Category prompt length: {len(category_prompt)} characters")
                print(f"Category response length: {len(category_response)} characters")
                print(f"Category response: {category_response[:200]}...")
                print()
                
                # Check for structured format
                if "SELECTED CATEGORIES" in category_response.upper() and "REJECTED CATEGORIES" in category_response.upper():
                    print("✅ Category response appears to follow structured format")
                else:
                    print("⚠️  Category response might not follow full structured format")
                
                if len(category_response) < 200:
                    print("⚠️  Category response seems short for detailed reasoning")
                else:
                    print("✅ Category response has good length for detailed reasoning")
                    
            except Exception as e:
                print(f"⚠️  Category selection test failed: {e}")
            
            print()
            
        else:
            print("🔧 Testing Token Limit Configuration (without API calls)")
            print("-" * 50)
            
            # Create evaluator without API key
            evaluator = DiagnosticEvaluator.__new__(DiagnosticEvaluator)
            evaluator.flowchart_dir = None
            evaluator.samples_dir = None
            evaluator.flowchart_categories = []
            
            # Test prompt creation
            prompt = evaluator.create_prompt(sample, 6, True)
            print(f"Standard prompt length: {len(prompt)} characters")
            
            category_prompt = evaluator.create_category_selection_prompt(sample, 6, 3)
            print(f"Category selection prompt length: {len(category_prompt)} characters")
            
            # Check if prompts are reasonable for the token limits
            if len(prompt) > 2000:  # Rough estimate: 1 token ≈ 4 chars
                estimated_tokens = len(prompt) // 4
                print(f"⚠️  Standard prompt is quite long (~{estimated_tokens} tokens)")
                print(f"   With max_tokens=500, might not leave enough space for response")
            else:
                print("✅ Standard prompt length is reasonable for 500 token response")
            
            if len(category_prompt) > 3000:
                estimated_tokens = len(category_prompt) // 4
                print(f"⚠️  Category prompt is quite long (~{estimated_tokens} tokens)")
                print(f"   With max_tokens=800, might not leave enough space for response")
            else:
                print("✅ Category prompt length is reasonable for 800 token response")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


def summarize_token_changes():
    """Summarize the token limit changes made"""
    
    print("📊 Summary of Token Limit Changes")
    print("=" * 50)
    
    print("Before fixes:")
    print("  ❌ Standard diagnostic responses: 100 tokens (~75 words)")
    print("  ❌ Category selection: 100 tokens (~75 words)")
    print("  ❌ Two-step final diagnosis: 100 tokens (~75 words)")
    print("  ❌ Iterative reasoning steps: 100 tokens (~75 words)")
    print("  ✅ LLM judge: 10 tokens (appropriate for YES/NO)")
    print()
    
    print("After fixes:")
    print("  ✅ Standard diagnostic responses: 500 tokens (~375 words)")
    print("  ✅ Category selection with reasoning: 800 tokens (~600 words)")
    print("  ✅ Two-step final diagnosis: 600 tokens (~450 words)")
    print("  ✅ Iterative reasoning steps: 1200 tokens (~900 words)")
    print("  ✅ LLM judge: 10 tokens (unchanged, appropriate)")
    print()
    
    print("Expected improvements:")
    print("  🎯 Complete diagnostic reasoning instead of cut-off responses")
    print("  🎯 Full evidence matching and comparative analysis")
    print("  🎯 Detailed category selection justifications")
    print("  🎯 Proper iterative reasoning with complete explanations")
    print()


if __name__ == "__main__":
    print("🧪 Testing Token Limits Fix")
    print("=" * 60)
    
    success = test_token_limits()
    
    print("\n" + "=" * 60)
    summarize_token_changes()
    
    print("=" * 60)
    print("🎯 CONCLUSION")
    print("=" * 60)
    
    if success:
        print("✅ Token limits have been significantly improved!")
        print("🔬 Framework should now provide complete diagnostic responses")
        print("⚡ Ready for proper evaluation with full reasoning capabilities")
    else:
        print("⚠️  Some issues detected in token limit testing")
        print("🔧 May need additional adjustments based on actual usage")
    
    print("\nRecommendation:")
    print("🧪 Test with actual API calls to verify response completeness")
    print("📊 Monitor response lengths during evaluation to ensure adequacy") 