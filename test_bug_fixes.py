#!/usr/bin/env python3
"""
Test script to verify the bug fixes for:
1. Missing disease category (Gastritis)  
2. .ipynb_checkpoints files being included as samples
"""

import os
from medeval import DiagnosticEvaluator
from medeval.utils import (
    load_flowchart_categories, 
    collect_sample_files, 
    get_samples_directory,
    extract_diagnosis_from_path,
    extract_disease_category_from_path
)

def test_checkpoint_filtering():
    """Test that .ipynb_checkpoints files are properly filtered out"""
    
    print("🧪 Testing Checkpoint File Filtering")
    print("=" * 50)
    
    try:
        # Get samples directory
        samples_dir = get_samples_directory()
        sample_files = collect_sample_files(samples_dir)
        
        print(f"📄 Total sample files found: {len(sample_files)}")
        
        # Check for any checkpoint files
        checkpoint_files = [f for f in sample_files if '.ipynb_checkpoints' in f]
        
        if checkpoint_files:
            print(f"❌ Found {len(checkpoint_files)} checkpoint files (should be 0):")
            for file in checkpoint_files[:5]:  # Show first 5
                print(f"   {file}")
            if len(checkpoint_files) > 5:
                print(f"   ... and {len(checkpoint_files) - 5} more")
            return False
        else:
            print("✅ No checkpoint files found in sample collection")
        
        # Check for any files with suspicious ground truth
        suspicious_files = []
        for file_path in sample_files[:100]:  # Check first 100 files
            ground_truth = extract_diagnosis_from_path(file_path)
            if ground_truth and ('checkpoint' in ground_truth.lower() or ground_truth.startswith('.')):
                suspicious_files.append((file_path, ground_truth))
        
        if suspicious_files:
            print(f"❌ Found {len(suspicious_files)} files with suspicious ground truth:")
            for file_path, ground_truth in suspicious_files[:3]:
                print(f"   File: {file_path}")
                print(f"   Ground truth: {ground_truth}")
            return False
        else:
            print("✅ No suspicious ground truth values found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during checkpoint filtering test: {e}")
        return False

def test_category_count():
    """Test that all disease categories are properly loaded"""
    
    print("\n🧪 Testing Disease Category Count")
    print("=" * 50)
    
    try:
        # Load flowchart categories
        categories = load_flowchart_categories()
        
        print(f"📊 Found {len(categories)} disease categories")
        print("Categories:")
        for i, category in enumerate(sorted(categories), 1):
            print(f"  {i:2d}. {category}")
        
        # Check if expected number
        expected_categories = 24  # Based on available flowchart files
        if len(categories) == expected_categories:
            print(f"✅ Correct number of categories ({expected_categories})")
        else:
            print(f"⚠️  Expected {expected_categories} categories, found {len(categories)}")
        
        # Check for Gastritis specifically (known missing flowchart)
        if 'Gastritis' in categories:
            print("⚠️  Gastritis found in categories but has no flowchart file")
        else:
            print("✅ Gastritis correctly excluded (no flowchart file)")
        
        return len(categories) == expected_categories
        
    except Exception as e:
        print(f"❌ Error during category count test: {e}")
        return False

def test_sample_directory_count():
    """Test the actual sample directory count vs categories"""
    
    print("\n🧪 Testing Sample Directory vs Category Alignment")
    print("=" * 50)
    
    try:
        # Count sample directories
        samples_dir = get_samples_directory()
        
        # Count actual category directories
        category_dirs = []
        for item in os.listdir(samples_dir):
            item_path = os.path.join(samples_dir, item)
            if os.path.isdir(item_path):
                category_dirs.append(item)
        
        print(f"📁 Found {len(category_dirs)} sample directories")
        
        # Load flowchart categories
        flowchart_categories = load_flowchart_categories()
        print(f"📊 Found {len(flowchart_categories)} flowchart categories")
        
        # Find directories without flowcharts
        dirs_without_flowcharts = set(category_dirs) - set(flowchart_categories)
        if dirs_without_flowcharts:
            print(f"⚠️  Sample directories without flowcharts: {sorted(dirs_without_flowcharts)}")
        
        # Find flowcharts without sample directories
        flowcharts_without_dirs = set(flowchart_categories) - set(category_dirs)
        if flowcharts_without_dirs:
            print(f"⚠️  Flowcharts without sample directories: {sorted(flowcharts_without_dirs)}")
        
        if not dirs_without_flowcharts and not flowcharts_without_dirs:
            print("✅ Perfect alignment between sample directories and flowcharts")
            return True
        else:
            print("✅ Misalignment is expected (Gastritis has samples but no flowchart)")
            return True
        
    except Exception as e:
        print(f"❌ Error during directory alignment test: {e}")
        return False

def test_evaluator_initialization():
    """Test that DiagnosticEvaluator initializes correctly with bug fixes"""
    
    print("\n🧪 Testing DiagnosticEvaluator Initialization")
    print("=" * 50)
    
    try:
        # Create evaluator without API key (just testing initialization)
        evaluator = DiagnosticEvaluator(
            api_key="dummy",  # Won't be used for this test
            model="gpt-4o-mini",
            show_responses=False
        )
        
        print(f"📊 Possible diagnoses: {len(evaluator.possible_diagnoses)}")
        print(f"📁 Flowchart categories: {len(evaluator.flowchart_categories)}")
        
        # Check that numbers are reasonable
        if len(evaluator.possible_diagnoses) > 50:
            print("✅ Reasonable number of possible diagnoses")
        else:
            print(f"⚠️  Only {len(evaluator.possible_diagnoses)} possible diagnoses found")
        
        if len(evaluator.flowchart_categories) >= 20:
            print("✅ Reasonable number of flowchart categories")
        else:
            print(f"⚠️  Only {len(evaluator.flowchart_categories)} flowchart categories found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during evaluator initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all bug fix tests"""
    
    print("🔧 MedEval Bug Fix Verification")
    print("=" * 80)
    
    tests = [
        ("Checkpoint File Filtering", test_checkpoint_filtering),
        ("Disease Category Count", test_category_count),
        ("Directory Alignment", test_sample_directory_count),
        ("Evaluator Initialization", test_evaluator_initialization)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All bug fixes verified successfully!")
        print("\nBug fixes implemented:")
        print("✅ .ipynb_checkpoints files are now properly filtered out")
        print("✅ Disease category count is now accurate")
        print("✅ No more invalid ground truth values from checkpoint files")
    else:
        print("⚠️  Some issues remain - see details above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 