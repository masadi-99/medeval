#!/usr/bin/env python3
"""
Test script for multi-model functionality
"""

import os
from medeval import DiagnosticEvaluator
from medeval.models import PREDEFINED_MODELS, create_model_provider

def test_model_loading():
    """Test loading different model providers"""
    
    print("🧪 Testing Multi-Model Functionality")
    print("=" * 50)
    
    # Show available predefined models
    print("📋 Available predefined models:")
    for name, config in PREDEFINED_MODELS.items():
        print(f"  {name}: {config.get('model_name', config.get('model', name))} ({config['provider']})")
    print()
    
    # Test OpenAI model (if API key available)
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("✅ Testing OpenAI provider...")
        try:
            evaluator = DiagnosticEvaluator(
                api_key=openai_key,
                model="gpt-4o-mini",
                provider="openai"
            )
            print(f"   OpenAI evaluator initialized successfully")
            print(f"   Model: {evaluator.model}")
            print(f"   Provider: {evaluator.provider_type}")
        except Exception as e:
            print(f"   ❌ OpenAI initialization failed: {e}")
    else:
        print("⚠️  OPENAI_API_KEY not set, skipping OpenAI test")
    
    print()
    
    # Test HuggingFace API model (if token available)
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if hf_token:
        print("✅ Testing HuggingFace API provider...")
        try:
            evaluator = DiagnosticEvaluator(
                model="Qwen/Qwen3-30B-A3B",
                provider="huggingface_api",
                huggingface_token=hf_token
            )
            print(f"   HuggingFace API evaluator initialized successfully")
            print(f"   Model: {evaluator.model}")
            print(f"   Provider: {evaluator.provider_type}")
        except Exception as e:
            print(f"   ❌ HuggingFace API initialization failed: {e}")
    else:
        print("⚠️  HUGGINGFACE_TOKEN not set, skipping HuggingFace API test")
    
    print()
    
    # Test predefined model loading
    print("✅ Testing predefined model configurations...")
    try:
        # This should work without API keys (will use first available provider)
        for model_name in ['gpt-4o-mini', 'qwen3-30b']:
            try:
                config = PREDEFINED_MODELS[model_name]
                print(f"   Model '{model_name}' config: {config}")
                
                if config['provider'] == 'openai' and not openai_key:
                    print(f"   Skipping {model_name} (no OpenAI API key)")
                    continue
                elif config['provider'] == 'huggingface':
                    print(f"   Skipping {model_name} (requires transformers library and GPU)")
                    continue
                
                # Only test if we have the required dependencies
                evaluator = DiagnosticEvaluator(
                    model=model_name,
                    api_key=openai_key,
                    huggingface_token=hf_token
                )
                print(f"   ✅ {model_name} initialized successfully")
                
            except Exception as e:
                print(f"   ⚠️  {model_name} initialization skipped: {e}")
                
    except Exception as e:
        print(f"   ❌ Predefined model test failed: {e}")
    
    print()
    
    # Test auto-detection
    print("✅ Testing provider auto-detection...")
    test_cases = [
        ("gpt-4o-mini", "should detect openai"),
        ("Qwen/Qwen3-30B-A3B", "should detect huggingface"),
        ("meta-llama/Meta-Llama-3-8B-Instruct", "should detect huggingface"),
    ]
    
    for model_name, description in test_cases:
        try:
            # Create evaluator without specifying provider
            evaluator = DiagnosticEvaluator(
                model=model_name,
                provider="auto",  # Let it auto-detect
                api_key=openai_key,
                huggingface_token=hf_token
            )
            print(f"   {model_name} -> {evaluator.provider_type} ✅")
        except Exception as e:
            print(f"   {model_name} -> failed: {e} ⚠️")
    
    print()
    print("🎯 Multi-model functionality test completed!")


def test_simple_query():
    """Test a simple query with available models"""
    
    print("\n🔬 Testing Simple Queries")
    print("=" * 30)
    
    # Simple test prompt
    test_prompt = "A 65-year-old patient presents with chest pain and shortness of breath. What are the key diagnostic considerations?"
    
    # Test with OpenAI if available
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("Testing OpenAI query...")
        try:
            evaluator = DiagnosticEvaluator(
                api_key=openai_key,
                model="gpt-4o-mini",
                show_responses=True
            )
            response = evaluator.query_llm(test_prompt)
            print(f"OpenAI Response: {response[:100]}...")
        except Exception as e:
            print(f"OpenAI query failed: {e}")
    
    print("\n✅ Query test completed!")


if __name__ == "__main__":
    test_model_loading()
    test_simple_query() 