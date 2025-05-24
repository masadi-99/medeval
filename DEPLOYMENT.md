# MedEval Deployment Guide

This guide will help you deploy the MedEval package on your server.

## Quick Deployment

### 1. Clone the Repository

```bash
git clone https://github.com/mohammadasadi/medeval.git
cd medeval
```

### 2. Install the Package

```bash
# Install in development mode (recommended)
pip install -e .

# Or install directly
pip install .
```

### 3. Set Up Environment Variables

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

### 4. Verify Installation

```bash
# Check commands are available
medeval --help
medeval-demo --help
medeval-show --help

# Test with sample data analysis
medeval-show
```

## Production Installation

### Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv medeval-env
source medeval-env/bin/activate  # On Windows: medeval-env\Scripts\activate

# Clone and install
git clone https://github.com/mohammadasadi/medeval.git
cd medeval
pip install -e .
```

### Using Docker (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

# Set environment variable for API key
ENV OPENAI_API_KEY=""

CMD ["medeval", "--help"]
```

Build and run:

```bash
docker build -t medeval .
docker run -e OPENAI_API_KEY="your-key" medeval medeval-show
```

## Usage Examples

### Basic Evaluation

```bash
# Quick demo (5 samples per test)
medeval-demo

# Small evaluation (50 samples)
medeval --api-key $OPENAI_API_KEY --max-samples 50 --output results_50.json

# Full evaluation (all samples)
medeval --api-key $OPENAI_API_KEY --output full_evaluation.json
```

### Research Experiments

```bash
# Information progression experiment
for inputs in {1..6}; do
    medeval --api-key $OPENAI_API_KEY \
        --num-inputs $inputs \
        --max-samples 100 \
        --output "results_${inputs}_inputs.json"
done

# Open vs closed diagnosis
medeval --api-key $OPENAI_API_KEY --max-samples 200 --output closed_diagnosis.json
medeval --api-key $OPENAI_API_KEY --no-list --max-samples 200 --output open_diagnosis.json
```

### Batch Processing

Create a script `run_evaluation.sh`:

```bash
#!/bin/bash

API_KEY="your-openai-api-key"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results_${TIMESTAMP}"

mkdir -p $OUTPUT_DIR

echo "Starting evaluation batch at $(date)"

# Test different input configurations
for inputs in 1 2 3 6; do
    echo "Running evaluation with $inputs inputs..."
    medeval --api-key $API_KEY \
        --num-inputs $inputs \
        --max-samples 100 \
        --output "$OUTPUT_DIR/results_${inputs}_inputs.json"
done

# Test with and without diagnosis list
echo "Running open diagnosis evaluation..."
medeval --api-key $API_KEY \
    --no-list \
    --max-samples 100 \
    --output "$OUTPUT_DIR/results_open_diagnosis.json"

echo "Evaluation batch completed at $(date)"
echo "Results saved in $OUTPUT_DIR/"
```

Make it executable and run:

```bash
chmod +x run_evaluation.sh
./run_evaluation.sh
```

## Custom Data

If you have your own diagnostic data:

### 1. Prepare Your Data Structure

```
your_data/
├── samples/
│   ├── Disease1/
│   │   ├── Subtype1/
│   │   │   ├── sample1.json
│   │   │   └── sample2.json
│   │   └── Subtype2/
│   └── Disease2/
└── flowcharts/
    ├── Disease1.json
    └── Disease2.json
```

### 2. Run Evaluation with Custom Data

```bash
medeval --api-key $OPENAI_API_KEY \
    --samples-dir /path/to/your_data/samples \
    --flowchart-dir /path/to/your_data/flowcharts \
    --max-samples 50
```

## Performance Optimization

### For Large-Scale Evaluations

1. **Use batch processing**: Split large evaluations into smaller chunks
2. **Monitor API rate limits**: OpenAI has rate limits that you need to respect
3. **Save intermediate results**: Use `--max-samples` to process in batches

Example batch script:

```bash
#!/bin/bash

API_KEY="your-key"
TOTAL_SAMPLES=1000
BATCH_SIZE=100
BATCHES=$((TOTAL_SAMPLES / BATCH_SIZE))

for i in $(seq 0 $((BATCHES-1))); do
    START=$((i * BATCH_SIZE))
    echo "Processing batch $((i+1))/$BATCHES (samples $START-$((START+BATCH_SIZE-1)))"
    
    # Note: You might need to implement sample range selection
    medeval --api-key $API_KEY \
        --max-samples $BATCH_SIZE \
        --output "batch_${i}_results.json"
    
    # Small delay to respect rate limits
    sleep 1
done
```

## Monitoring and Logging

### Enable Verbose Logging

```bash
# Set up logging
export PYTHONPATH="${PYTHONPATH}:/path/to/medeval"

# Run with Python logging
python -c "
import logging
logging.basicConfig(level=logging.INFO)
from medeval import DiagnosticEvaluator
evaluator = DiagnosticEvaluator('your-key')
results = evaluator.evaluate_dataset(max_samples=10)
print(f'Accuracy: {results[\"overall_metrics\"][\"accuracy\"]:.3f}')
"
```

### Results Analysis

```python
import json
import pandas as pd

# Load results
with open('evaluation_results.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(results['detailed_results'])

# Analyze performance by diagnosis
accuracy_by_diagnosis = df.groupby('ground_truth')['correct'].mean()
print("Accuracy by diagnosis:")
print(accuracy_by_diagnosis.sort_values(ascending=False))

# Show confusion patterns
errors = df[df['correct'] == False]
print("\nMost common errors:")
print(errors.groupby(['ground_truth', 'predicted_matched']).size().sort_values(ascending=False).head(10))
```

## Troubleshooting

### Common Issues

1. **Package not found**: Make sure you installed with `pip install -e .`
2. **Data not found**: Ensure you have the data files available locally
3. **API errors**: Check your OpenAI API key and rate limits
4. **Memory issues**: Use smaller batch sizes with `--max-samples`

### Environment Issues

```bash
# Check installation
pip list | grep medeval

# Check commands
which medeval
which medeval-demo
which medeval-show

# Check data files
python -c "from medeval.utils import get_data_path; print(get_data_path())"
```

### Getting Help

- Check command help: `medeval --help`
- Test with demo: `medeval-demo`
- Analyze data: `medeval-show`
- Check installation: `pip show medeval`

## Security Considerations

1. **API Key Security**: Never commit API keys to version control
2. **Environment Variables**: Use `.env` files or environment variables
3. **Data Privacy**: Ensure medical data complies with privacy regulations
4. **Rate Limiting**: Implement proper rate limiting for production use

## Scaling for Production

For high-throughput production use:

1. **Asynchronous Processing**: Consider implementing async API calls
2. **Caching**: Cache diagnosis lists and frequently used data
3. **Database Integration**: Store results in a database for analysis
4. **Monitoring**: Implement proper monitoring and alerting
5. **Error Handling**: Add robust error handling and retry logic

## Support

For issues or questions:
- Create an issue on GitHub
- Check the documentation in `README.md`
- Review the examples in this deployment guide 