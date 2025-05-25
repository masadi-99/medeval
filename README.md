# MedEval: LLM Diagnostic Reasoning Evaluator

A comprehensive evaluation framework for testing Large Language Models (LLMs) on medical diagnostic reasoning tasks using the MIMIC-IV-Ext-DiReCT dataset.

## Overview

This framework evaluates LLMs on their ability to provide primary discharge diagnoses based on clinical information. It supports configurable inputs, provides evaluation metrics, and allows comparison between different scenarios.

## Features

- **Multi-Model Support**: OpenAI models (GPT-4, GPT-3.5) and Hugging Face models (Qwen3-30B, Llama3, Mistral, etc.)
- **Local and API Inference**: Run models locally or via APIs for flexibility and cost control
- **Thinking Mode Support**: Special support for Qwen3's thinking/non-thinking modes for enhanced reasoning
- **Configurable Input Fields**: Choose which clinical information to provide (1-6 inputs)
- **Diagnosis List Toggle**: Option to provide or withhold the list of possible diagnoses
- **Two-Step Reasoning**: Advanced diagnostic reasoning with category selection followed by final diagnosis
- **Iterative Step-by-Step Reasoning**: Follow diagnostic flowcharts step-by-step to reach final diagnosis
- **LLM Judge**: Use an LLM to evaluate diagnostic equivalence (handles different wording/synonyms)
- **Response Inspection**: Option to show actual LLM responses during evaluation
- **Comprehensive Metrics**: Accuracy, precision, recall, and F1-score
- **Per-class Analysis**: Detailed performance metrics for each diagnosis
- **Disease Category Metrics**: Performance breakdown by medical specialty/disease category
- **Concurrent Processing**: Fast evaluation with rate-limited concurrent API calls
- **Flexible Evaluation**: Support for different models and sample sizes
- **Easy Installation**: Install as a Python package with pip
- **Environment Integration**: Uses standard environment variables for API keys

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

### Optional Dependencies

For Hugging Face models (local inference):
```bash
pip install transformers torch accelerate
```

For async operations with Hugging Face API:
```bash
pip install aiohttp
```

### Verify Installation

```bash
# Check if commands are available
medeval --help
medeval-demo --help
medeval-show --help

# List available predefined models
medeval --model list
```

## Quick Start

### 1. Set up your API keys

For OpenAI models:
```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

For Hugging Face API models:
```bash
export HUGGINGFACE_TOKEN='your-huggingface-token-here'
```

### 2. Run a quick demo

```bash
# Demo with OpenAI GPT-4o-mini (default)
medeval-demo

# Demo with Qwen3-30B (local inference) - requires GPU and transformers
medeval-demo --model qwen3-30b --show-responses

# Demo with Qwen3 via Hugging Face API
medeval-demo --model Qwen/Qwen3-30B-A3B --provider huggingface

# Demo with custom model
medeval-demo --model meta-llama/Meta-Llama-3-8B-Instruct --provider huggingface
```

### 3. Analyze the dataset

```bash
medeval-show
```

### 4. Run evaluations with different models

```bash
# OpenAI models
medeval --model gpt-4o-mini --max-samples 50
medeval --model gpt-4o --max-samples 20

# Qwen3 with thinking mode (local inference)
medeval --model qwen3-30b --max-samples 20 --thinking-mode

# Qwen3 without thinking mode (faster)
medeval --model qwen3-30b-no-thinking --max-samples 50

# Custom Hugging Face model
medeval --model Qwen/Qwen3-30B-A3B --provider huggingface --max-samples 20

# Large-scale evaluation with concurrent processing
medeval --model qwen3-30b --concurrent --max-samples 200
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

## Supported Models

### OpenAI Models
- **gpt-4o**: Latest GPT-4 optimized model
- **gpt-4o-mini**: Faster, cost-effective GPT-4 model (default)
- **gpt-4-turbo**: High-performance GPT-4 variant
- **gpt-3.5-turbo**: Cost-effective option

### Hugging Face Models (Local Inference)

#### Qwen3 Series (Recommended)
- **qwen3-30b**: [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) with thinking mode enabled
- **qwen3-30b-no-thinking**: Same model with thinking mode disabled for faster inference

**Special Features of Qwen3:**
- **Thinking Mode**: Model generates internal reasoning (`<think>...</think>`) before final answer
- **Medical Reasoning**: Excellent performance on complex diagnostic reasoning tasks
- **Context Length**: 32,768 tokens natively (extendable to 131,072 with YaRN)
- **Optimized Parameters**: Automatic temperature/sampling settings based on mode

#### Other Popular Models
- **llama3-8b**: Meta-Llama-3-8B-Instruct
- **mistral-7b**: Mistral-7B-Instruct-v0.3
- **Custom Models**: Any Hugging Face model with chat template support

### Hugging Face API Models
Access any Hugging Face model via their Inference API (requires HuggingFace token):
```bash
medeval --model Qwen/Qwen3-30B-A3B --provider huggingface_api
```

### Model Selection Guide

**For Best Performance:**
- **Qwen3-30B** (thinking mode): Best for complex diagnostic reasoning
- **GPT-4o**: Excellent general performance with good speed

**For Speed/Cost:**
- **Qwen3-30B** (non-thinking): Fast inference with good accuracy
- **GPT-4o-mini**: Fast and cost-effective

**For Local/Private Use:**
- Any Hugging Face model with local inference
- No API calls, complete data privacy

**Hardware Requirements for Local Models:**
- **Qwen3-30B**: Requires 16-24GB GPU memory (30.5B parameters, MoE architecture)
- **Llama3-8B**: Requires 8-12GB GPU memory
- **Mistral-7B**: Requires 6-10GB GPU memory

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

# Two-step diagnostic reasoning (select categories first, then diagnose)
medeval --two-step --max-samples 50

# Two-step with custom number of categories to select
medeval --two-step --num-categories 5 --max-samples 50

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
| **Model Selection** | | |
| `--model` | Model name or predefined model key | gpt-4o-mini |
| `--provider` | Model provider (auto, openai, huggingface, huggingface_api) | auto |
| **Authentication** | | |
| `--api-key` | OpenAI API key | Uses `OPENAI_API_KEY` env var |
| `--huggingface-token` | HuggingFace token | Uses `HUGGINGFACE_TOKEN` env var |
| **Model Configuration** | | |
| `--device` | Device for local models (auto, cpu, cuda) | auto |
| `--torch-dtype` | Torch dtype (auto, float16, bfloat16) | auto |
| `--thinking-mode` | Enable thinking mode for compatible models | True |
| `--no-thinking-mode` | Disable thinking mode | False |
| **Evaluation Options** | | |
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
| **Reasoning Modes** | | |
| `--two-step` | Use two-step diagnostic reasoning | False |
| `--num-categories` | Number of categories to select in two-step mode | 3 |
| `--iterative` | Use iterative step-by-step reasoning following flowcharts | False |
| `--max-reasoning-steps` | Maximum number of reasoning steps in iterative mode | 5 |
| **Concurrency** | | |
| `--concurrent` | Use concurrent processing for faster evaluation | False |
| `--max-concurrent` | Maximum number of concurrent API calls | 10 |

## Two-Step Diagnostic Reasoning

The Two-Step Reasoning feature implements a sophisticated diagnostic approach that mimics real clinical decision-making:

### How it Works:

**Step 1: Category Selection**
- LLM receives clinical data and a list of 25 disease categories (from flowcharts)
- Selects the k most likely disease categories (default k=3)
- Evaluated on whether the correct disease category is among selected categories

**Step 2: Final Diagnosis**
- LLM receives:
  - Original clinical data
  - Detailed flowcharts for the k selected categories
  - Complete list of possible diagnoses
- Makes final diagnostic decision using the flowchart guidance

### Benefits:
- **Structured Reasoning**: Mirrors clinical decision-making process
- **Guided Decision Making**: Flowcharts provide structured diagnostic pathways
- **Improved Accuracy**: Focused attention on relevant disease categories
- **Interpretability**: Clear intermediate reasoning steps

### Usage:
```bash
# Basic two-step reasoning (selects 3 categories)
medeval --two-step --max-samples 50

# Select more categories for broader consideration
medeval --two-step --num-categories 5 --max-samples 50

# Two-step with detailed response inspection
medeval --two-step --show-responses --max-samples 10
```

### Output Metrics:
- **Category Selection Accuracy**: How often the correct disease category is selected
- **Final Diagnosis Accuracy**: Overall diagnostic accuracy
- **Per-Category Performance**: Breakdown by disease category

## Iterative Step-by-Step Reasoning

The Iterative Step-by-Step Reasoning feature implements the most sophisticated diagnostic approach that explicitly follows diagnostic flowcharts node by node, mimicking how clinicians use structured diagnostic criteria.

### How it Works:

**Step 1: Enhanced Category Selection**
- When selecting multiple categories (k > 1), LLM must provide detailed reasoning for each choice and rejection
- Must match specific patient findings to typical presentations of each category
- Required to explain why certain categories are supported/rejected by the available evidence
- Encourages reconsideration of initial diagnostic impressions based on systematic analysis

**Step 2: Evidence-Based Flowchart Traversal**
- LLM starts at the root of the selected flowchart(s)
- At each node, LLM receives both patient data and specific clinical criteria for each diagnostic option
- Must match specific patient observations to specific clinical criteria (no general medical knowledge)
- Provides structured reasoning: EVIDENCE MATCHING (comparing patient findings to diagnostic criteria), COMPARATIVE ANALYSIS (why one option over others), DECISION (chosen path), and RATIONALE (evidence-based justification)
- When multiple options exist, must explain choice and rejection of alternatives based solely on available evidence
- Continues until reaching a leaf node (final diagnosis)
- Each step captures detailed clinical reasoning tied to specific patient findings

### Benefits:
- **Enhanced Category Selection**: Forces systematic consideration of each category with evidence-based justification
- **Evidence-Only Reasoning**: Prevents reliance on general medical knowledge, focusing on available patient data
- **Explicit Reasoning**: Every diagnostic step is recorded and interpretable with specific evidence matching
- **Clinical Evidence Matching**: LLM must justify each step by connecting patient findings to diagnostic criteria
- **Comparative Decision Making**: When multiple options exist, must explain choice rationale and rejection reasoning
- **Structured Decision Making**: Follows established clinical decision trees with detailed evidence-based reasoning
- **Path Validation**: Can evaluate if the reasoning path used correct disease categories and evidence
- **Educational Value**: Shows how diagnostic criteria are applied step-by-step with specific patient evidence
- **Error Analysis**: Identifies where in the reasoning process errors occur and why, with evidence trail
- **Rigorous Methodology**: Ensures reasoning is grounded in available data rather than general medical assumptions

### Usage:
```bash
# Basic iterative reasoning (selects 3 categories, max 5 steps)
medeval --iterative --max-samples 20

# Custom parameters
medeval --iterative --num-categories 5 --max-reasoning-steps 7 --max-samples 50

# With detailed reasoning inspection
medeval --iterative --show-responses --max-samples 5
```

### Output Metrics:
- **Category Selection Accuracy**: How often the correct disease category is selected
- **Reasoning Path Accuracy**: How often the correct disease category is used in the reasoning path
- **Final Diagnosis Accuracy**: Overall diagnostic accuracy
- **Average Reasoning Steps**: Average number of steps taken to reach diagnosis
- **Per-Category Performance**: Breakdown by disease category

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
6. **Reasoning Approaches**: How does two-step reasoning compare to direct diagnosis?
7. **Category Selection**: How well can LLMs identify relevant disease categories?
8. **Iterative Reasoning**: How does step-by-step flowchart traversal affect diagnostic accuracy?
9. **Reasoning Path Analysis**: Do models follow diagnostically sound reasoning paths?
10. **Thinking Mode Impact**: How does Qwen3's thinking mode affect diagnostic performance?
11. **Local vs API Models**: Performance comparison between local and API-based inference?

## Model Usage Examples

### OpenAI Models
```bash
# GPT-4o for high accuracy
medeval --model gpt-4o --max-samples 100

# GPT-4o-mini for faster evaluation
medeval --model gpt-4o-mini --max-samples 500 --concurrent

# Compare OpenAI models
medeval --model gpt-4o --max-samples 50 --output gpt4o_results.json
medeval --model gpt-4o-mini --max-samples 50 --output gpt4o_mini_results.json
```

### Qwen3 Models (Local Inference)
```bash
# Qwen3 with thinking mode (best for complex reasoning)
medeval --model qwen3-30b --thinking-mode --max-samples 50 --show-responses

# Qwen3 without thinking mode (faster)
medeval --model qwen3-30b-no-thinking --max-samples 100

# Qwen3 with iterative reasoning (most sophisticated)
medeval --model qwen3-30b --iterative --thinking-mode --max-samples 20

# Qwen3 concurrent evaluation
medeval --model qwen3-30b --concurrent --max-concurrent 5 --max-samples 200

# Custom device/precision settings
medeval --model qwen3-30b --device cuda:0 --torch-dtype bfloat16 --max-samples 50
```

### Hugging Face API Models
```bash
# Access any Hugging Face model via API
export HUGGINGFACE_TOKEN='your-token-here'

medeval --model Qwen/Qwen3-30B-A3B --provider huggingface_api --max-samples 50
medeval --model meta-llama/Meta-Llama-3-8B-Instruct --provider huggingface_api --max-samples 50
medeval --model mistralai/Mistral-7B-Instruct-v0.3 --provider huggingface_api --max-samples 50
```

### Custom Model Configurations
```bash
# Load any Hugging Face model locally
medeval --model microsoft/DialoGPT-medium --provider huggingface --max-samples 20

# Custom model with specific settings
medeval --model Qwen/Qwen3-30B-A3B --provider huggingface \
        --device cuda:1 --torch-dtype float16 --thinking-mode --max-samples 30

# Model comparison study
for model in qwen3-30b gpt-4o-mini llama3-8b; do
    medeval --model $model --max-samples 100 --output "${model}_results.json"
done
```

### Thinking Mode Evaluation
```bash
# Compare thinking vs non-thinking modes
medeval --model qwen3-30b --thinking-mode --max-samples 50 \
        --output qwen3_thinking_results.json

medeval --model qwen3-30b --no-thinking-mode --max-samples 50 \
        --output qwen3_no_thinking_results.json

# Show thinking content during evaluation
medeval --model qwen3-30b --thinking-mode --show-responses --max-samples 5
```

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

### Experiment 4: Two-Step vs. Direct Reasoning

```bash
# Direct reasoning
medeval --max-samples 200 \
    --output results_direct.json

# Two-step reasoning
medeval --two-step \
    --max-samples 200 \
    --output results_two_step.json

# Two-step with different category counts
for k in 2 3 5; do
    medeval --two-step --num-categories $k \
        --max-samples 100 \
        --output "results_two_step_k${k}.json"
done
```

### Experiment 5: Iterative Step-by-Step Reasoning

```bash
# Compare reasoning approaches
medeval --max-samples 100 \
    --output results_direct.json

medeval --two-step --max-samples 100 \
    --output results_two_step.json

medeval --iterative --max-samples 50 \
    --output results_iterative.json

# Iterative reasoning with different parameters
for steps in 3 5 7; do
    medeval --iterative --max-reasoning-steps $steps \
        --max-samples 50 \
        --output "results_iterative_${steps}steps.json"
done

# Different category selection strategies
for k in 2 3 5; do
    medeval --iterative --num-categories $k \
        --max-samples 50 \
        --output "results_iterative_k${k}.json"
done
```

### Experiment 6: Large-Scale Concurrent Evaluation

```bash
# Large-scale studies with concurrent processing
medeval --concurrent --max-samples 500 \
    --output full_dataset_standard.json

medeval --two-step --concurrent --max-samples 500 \
    --output full_dataset_two_step.json

medeval --iterative --concurrent --max-samples 200 \
    --output full_dataset_iterative.json

# Performance comparison: sequential vs concurrent
time medeval --max-samples 100 --output sequential.json
time medeval --concurrent --max-samples 100 --output concurrent.json

# Large-scale parameter sweep
for inputs in {1..6}; do
    for reasoning in "" "--two-step" "--iterative"; do
        medeval --concurrent --num-inputs $inputs $reasoning \
            --max-samples 200 \
            --output "large_scale_${inputs}inputs_${reasoning/--/}.json"
    done
done
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

## Concurrent Processing for Large-Scale Evaluation

For research with hundreds or thousands of samples, the framework supports **concurrent processing** with automatic rate limiting to maximize throughput while respecting OpenAI API limits.

### Performance Benefits:
- **10-50x Faster**: Process multiple samples simultaneously instead of sequentially
- **Rate Limit Compliance**: Automatic rate limiting stays within OpenAI's 500 requests/minute limit
- **Batch Processing**: Intelligent batching prevents API overload
- **Progress Tracking**: Real-time progress updates with ETA estimates
- **Error Handling**: Graceful handling of failed requests without stopping evaluation

### Usage:
```bash
# Enable concurrent processing (recommended for >50 samples)
medeval --concurrent --max-samples 500

# Adjust concurrency level (default: 10 concurrent requests)
medeval --concurrent --max-concurrent 15 --max-samples 500

# Concurrent iterative reasoning
medeval --iterative --concurrent --max-samples 200

# Large-scale evaluation with progress tracking
medeval --concurrent --max-concurrent 20 --output large_study.json
```

### Performance Expectations:
- **Standard Mode**: ~3-5 samples/second (vs 0.5-1 samples/second sequential)
- **Two-Step Mode**: ~2-3 samples/second (vs 0.3-0.5 samples/second sequential)  
- **Iterative Mode**: ~1-2 samples/second (vs 0.1-0.3 samples/second sequential)
- **500 samples**: ~3-5 minutes (vs 15-30 minutes sequential)

### Rate Limit Management:
- Default: 450 requests/minute (leaves buffer below 500 limit)
- Automatically throttles to prevent hitting limits
- Handles temporary rate limit errors gracefully
- Balances speed with API compliance
