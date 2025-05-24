# MedEval: LLM Diagnostic Reasoning Evaluator

A comprehensive evaluation framework for testing Large Language Models (LLMs) on medical diagnostic reasoning tasks using the MIMIC-IV-Ext-DiReCT dataset.

## Overview

This framework evaluates LLMs on their ability to provide primary discharge diagnoses based on clinical information. It supports configurable inputs, provides evaluation metrics, and allows comparison between different scenarios.

## Features

- **Configurable Input Fields**: Choose which clinical information to provide (1-6 inputs)
- **Diagnosis List Toggle**: Option to provide or withhold the list of possible diagnoses
- **LLM Judge**: Use an LLM to evaluate diagnostic equivalence (handles different wording/synonyms)
- **Response Inspection**: Option to show actual LLM responses during evaluation
- **Comprehensive Metrics**: Accuracy, precision, recall, and F1-score
- **Per-class Analysis**: Detailed performance metrics for each diagnosis
- **Flexible Evaluation**: Support for different models and sample sizes
- **Easy Installation**: Install as a Python package with pip
- **Environment Integration**: Uses standard OpenAI environment variables

## Installation

### From Git Repository

```bash
# Clone the repository
git clone https://github.com/mohammadasadi/medeval.git
cd medeval

# Install the package
pip install -e .

# Or install directly from git
pip install git+https://github.com/mohammadasadi/medeval.git
```

### Verify Installation

```bash
# Check if commands are available
medeval --help
medeval-demo --help
medeval-show --help
```

## Quick Start

### 1. Set up your OpenAI API key

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

### 2. Run a quick demo

```bash
medeval-demo
```

### 3. Analyze the dataset

```bash
medeval-show
```

### 4. Run a full evaluation

```bash
medeval --max-samples 50
```

## Dataset Structure

The framework works with the MIMIC-IV-Ext-DiReCT dataset which contains:

### Clinical Input Fields
1. **Chief Complaint**: Primary reason for admission
2. **History of Present Illness**: Detailed description of current condition
3. **Past Medical History**: Previous medical conditions and treatments
4. **Family History**: Relevant family medical history
5. **Physical Examination**: Clinical findings from physical exam
6. **Laboratory Results**: Lab values and diagnostic test results

### Possible Diagnoses
The framework supports 61 primary discharge diagnoses across 25 medical categories:
- Acute Coronary Syndrome (NSTEMI, STEMI, UA)
- Heart Failure (HFrEF, HFmrEF, HFpEF)
- Stroke (Ischemic, Hemorrhagic)
- Diabetes (Type I, Type II)
- And many more...

## Usage

### Command Line Interface

The package provides three main commands:

#### 1. `medeval` - Main evaluation command

```bash
# Basic evaluation (uses OPENAI_API_KEY environment variable)
medeval --max-samples 50

# Evaluate with limited clinical information
medeval --num-inputs 2 --max-samples 100

# Evaluate without providing diagnosis list (open-ended)
medeval --no-list --max-samples 100

# Show actual LLM responses during evaluation
medeval --show-responses --max-samples 10

# Use exact string matching instead of LLM judge
medeval --no-llm-judge --max-samples 50

# Full evaluation with all samples
medeval --output full_evaluation.json
```

#### 2. `medeval-demo` - Quick demonstration

```bash
# Run demo (uses OPENAI_API_KEY environment variable)
medeval-demo

# Run demo with explicit API key
medeval-demo --api-key YOUR_API_KEY

# Show LLM responses during demo
medeval-demo --show-responses
```

#### 3. `medeval-show` - Dataset analysis

```bash
# Show sample data and statistics
medeval-show
```

### Python API

You can also use the package programmatically:

```python
from medeval import DiagnosticEvaluator

# Initialize evaluator (uses OPENAI_API_KEY environment variable)
evaluator = DiagnosticEvaluator(
    api_key="your-api-key",  # or use os.getenv('OPENAI_API_KEY')
    use_llm_judge=True,      # Enable LLM judge (default)
    show_responses=False     # Show LLM responses (default: False)
)

# Run evaluation
results = evaluator.evaluate_dataset(
    num_inputs=6,
    provide_diagnosis_list=True,
    max_samples=100
)

# Print results
print(f"Accuracy: {results['overall_metrics']['accuracy']:.3f}")

# Save results
evaluator.save_results(results, "my_results.json")
```

## Command Line Arguments

### `medeval` command

| Argument | Description | Default |
|----------|-------------|---------|
| `--api-key` | OpenAI API key | Uses `OPENAI_API_KEY` env var |
| `--model` | Model to use | gpt-4o-mini |
| `--num-inputs` | Number of input fields (1-6) | 6 |
| `--provide-list` | Provide diagnosis list | True |
| `--no-list` | Don't provide diagnosis list | False |
| `--max-samples` | Maximum samples to evaluate | All |
| `--output` | Output JSON file | evaluation_results.json |
| `--samples-dir` | Custom samples directory | Package data |
| `--flowchart-dir` | Custom flowchart directory | Package data |
| `--show-responses` | Show actual LLM responses | False |
| `--use-llm-judge` | Use LLM as judge for evaluation | True |
| `--no-llm-judge` | Use exact string matching | False |

## LLM Judge Feature

The LLM Judge is a key feature that significantly improves evaluation accuracy, especially when diagnosis lists are not provided. Instead of exact string matching, it uses the same LLM to determine if two diagnoses refer to the same medical condition.

### Benefits:
- **Handles Synonyms**: "Heart Attack" vs "Myocardial Infarction"
- **Medical Abbreviations**: "MI" vs "Myocardial Infarction"  
- **Different Phrasing**: "Type 2 Diabetes" vs "Diabetes Mellitus Type II"
- **Clinical Equivalents**: "CHF" vs "Congestive Heart Failure"

### Usage:
```bash
# LLM Judge enabled (default)
medeval --max-samples 50

# Disable LLM Judge (exact matching only)
medeval --no-llm-judge --max-samples 50
```

## Output Format

The evaluation produces a JSON file with:

```json
{
  "overall_metrics": {
    "accuracy": 0.85,
    "precision": 0.83,
    "recall": 0.85,
    "f1": 0.84,
    "num_samples": 100
  },
  "per_class_metrics": {
    "NSTEMI": {
      "precision": 0.90,
      "recall": 0.85,
      "f1": 0.87,
      "support": 20
    }
  },
  "configuration": {
    "num_inputs": 6,
    "provide_diagnosis_list": true,
    "model": "gpt-4o-mini",
    "use_llm_judge": true,
    "show_responses": false
  },
  "detailed_results": [
    {
      "sample_path": "...",
      "ground_truth": "NSTEMI",
      "predicted_raw": "Non-ST elevation myocardial infarction",
      "predicted_matched": "NSTEMI",
      "correct": true,
      "evaluation_method": "llm_judge"
    }
  ]
}
```

## Examples of Research Questions

This framework enables investigation of various research questions:

1. **Information Sufficiency**: How does diagnostic accuracy change with the amount of clinical information?
2. **Open vs. Closed Diagnosis**: How does performance differ when the model has access to possible diagnoses vs. open-ended diagnosis?
3. **Model Comparison**: How do different LLMs perform on diagnostic tasks?
4. **Error Analysis**: What types of diagnostic errors do LLMs make most frequently?
5. **Evaluation Methods**: How does LLM judge compare to exact string matching?

## Experimental Design Examples

### Experiment 1: Information Progression
Compare accuracy as more clinical information is provided:

```bash
for inputs in {1..6}; do
    medeval --num-inputs $inputs \
        --max-samples 100 \
        --output "results_${inputs}_inputs.json"
done
```

### Experiment 2: Closed vs. Open Diagnosis

```bash
# With diagnosis list
medeval --max-samples 200 \
    --output results_with_list.json

# Without diagnosis list  
medeval --no-list \
    --max-samples 200 \
    --output results_no_list.json
```

### Experiment 3: LLM Judge vs. Exact Matching

```bash
# With LLM judge
medeval --no-list \
    --max-samples 100 \
    --output results_llm_judge.json

# With exact matching
medeval --no-list --no-llm-judge \
    --max-samples 100 \
    --output results_exact_match.json
```

## Custom Data

You can use your own data by specifying custom directories:

```bash
medeval --samples-dir /path/to/your/samples \
    --flowchart-dir /path/to/your/flowcharts
```

## Development

To contribute to the package:

```bash
# Clone and install in development mode
git clone https://github.com/mohammadasadi/medeval.git
cd medeval
pip install -e .

# Run tests
python -m pytest tests/

# Build package
python -m build
```

## Dataset Citation

When using this framework, please cite the original dataset:

```
Wang, B., Chang, J., & Qian, Y. (2025). MIMIC-IV-Ext-DiReCT (version 1.0.0). 
PhysioNet. https://doi.org/10.13026/yf96-kc87
```

## Contributing

Contributions are welcome! Possible enhancements:
- Support for additional models (Anthropic, Gemini, etc.)
- Advanced prompt engineering techniques
- Reasoning explanation evaluation
- Multi-step diagnostic reasoning
- Integration with other medical datasets
- Web interface for evaluation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting

### Common Issues

1. **Module not found errors**: Make sure you installed the package correctly with `pip install -e .`

2. **Data not found**: The package includes sample data, but if you're using custom data, ensure the paths are correct

3. **API key issues**: Make sure your OpenAI API key is valid and has sufficient credits
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

4. **Permission errors**: On some systems, you might need to use `pip install --user -e .`

5. **LLM Judge errors**: If the LLM judge fails, it falls back to exact matching automatically

### Getting Help

- Check the command help: `medeval --help`
- Run the demo to verify setup: `medeval-demo`
- Analyze your data: `medeval-show`
- Open an issue on GitHub for bugs or feature requests
