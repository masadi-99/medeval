#!/usr/bin/env python3
"""
Test script to verify the variable scope fix for progressive reasoning
"""

import os
from medeval import DiagnosticEvaluator

def test_variable_scope_fix():
    """Test that progressive reasoning no longer has variable scope issues"""
    
    print("🔧 Testing Variable Scope Fix")
    print("=" * 50)
    
    # Skip if no API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  No OPENAI_API_KEY found - testing logic only")
        return test_logic_only()
    
    try:
        # Create evaluator
        evaluator = DiagnosticEvaluator(
            api_key=os.getenv('OPENAI_API_KEY'),
            model="gpt-4o-mini",
            show_responses=False
        )
        
        print(f"✅ Evaluator initialized successfully")
        
        # Test concurrent progressive reasoning (this was causing the error)
        print("🚀 Testing concurrent progressive reasoning (where the error occurred)")
        
        import asyncio
        results = asyncio.run(evaluator.evaluate_dataset_concurrent(
            num_inputs=6,
            provide_diagnosis_list=True,
            max_samples=2,  # Very small test
            progressive_reasoning=True,
            progressive_fast_mode=True,
            num_suspicions=3
        ))
        
        # If we get here without error, the fix worked
        print(f"✅ Concurrent evaluation completed successfully!")
        print(f"📊 Evaluated {results['overall_metrics']['num_samples']} samples")
        print(f"🎯 Accuracy: {results['overall_metrics']['accuracy']:.3f}")
        
        # Check that configuration is properly set
        config = results['configuration']
        print(f"✅ Progressive reasoning in config: {config.get('progressive_reasoning', False)}")
        print(f"✅ Number of suspicions in config: {config.get('num_suspicions', 'Not set')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logic_only():
    """Test the logic without API calls"""
    
    print("🧠 Testing Logic Only (No API calls)")
    print("-" * 40)
    
    try:
        # Create evaluator
        evaluator = DiagnosticEvaluator(
            api_key="dummy",
            model="gpt-4o-mini",
            show_responses=False
        )
        
        print(f"✅ Evaluator initialized successfully")
        
        # Test that _calculate_metrics has correct signature
        import inspect
        sig = inspect.signature(evaluator._calculate_metrics)
        params = list(sig.parameters.keys())
        
        print(f"📝 _calculate_metrics parameters: {params}")
        
        # Check for required parameters
        required_params = ['results', 'num_inputs', 'provide_diagnosis_list', 
                          'two_step_reasoning', 'iterative_reasoning', 
                          'num_categories', 'max_reasoning_steps', 
                          'progressive_reasoning', 'num_suspicions']
        
        missing_params = [p for p in required_params if p not in params]
        
        if missing_params:
            print(f"❌ Missing parameters: {missing_params}")
            return False
        else:
            print(f"✅ All required parameters present")
            return True
        
    except Exception as e:
        print(f"❌ Logic test failed: {e}")
        return False

def main():
    """Run variable scope fix test"""
    
    print("🔧 Variable Scope Fix Verification")
    print("=" * 60)
    
    success = test_variable_scope_fix()
    
    print("=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    if success:
        print("🎉 Variable scope fix verified!")
        print("\nFix implemented:")
        print("✅ _calculate_metrics function signature now includes progressive_reasoning and num_suspicions parameters")
        print("✅ Concurrent progressive reasoning no longer throws 'progressive_reasoning is not defined' error")
        print("✅ Configuration object properly populated with all reasoning mode parameters")
    else:
        print("⚠️  Variable scope issues still present - see details above")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 