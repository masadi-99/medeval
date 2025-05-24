"""
MedEval: LLM Diagnostic Reasoning Evaluator

A comprehensive evaluation framework for testing Large Language Models (LLMs) 
on medical diagnostic reasoning tasks using the MIMIC-IV-Ext-DiReCT dataset.
"""

__version__ = "0.1.0"
__author__ = "Mohammad Asadi"
__email__ = "your.email@example.com"

from .evaluator import DiagnosticEvaluator
from .utils import load_possible_diagnoses, extract_diagnosis_from_path

__all__ = [
    "DiagnosticEvaluator",
    "load_possible_diagnoses", 
    "extract_diagnosis_from_path"
] 