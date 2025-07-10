# DeepSeek LLM Text Completion Example

This example demonstrates how to use the DeepSeek LLM 7B Base model for text completion tasks.

## Requirements

- Python 3.8+
- CUDA-compatible GPU with at least 16GB VRAM (recommended)
- Required Python packages (see requirements.txt)

## Installation

1. Install the required packages:
```bash
python -m pip install -r requirements.txt
```

2. Run the example:
```bash
python deepseek_completion.py
```

## Model Information

- **Model**: deepseek-ai/deepseek-llm-67b-base
- **Parameters**: 67 billion
- **Precision**: bfloat16 for memory efficiency
- **Context Length**: 4096 tokens

## Features

- Automatic device mapping for optimal GPU utilization
- Configurable generation parameters
- Error handling and troubleshooting guidance
- Clean output formatting

## Customization

You can modify the generation parameters in the script:
- `max_new_tokens`: Maximum number of tokens to generate
- `temperature`: Controls randomness (0.1-1.0)
- `top_p`: Nucleus sampling parameter
- `repetition_penalty`: Reduces repetitive text

## Memory Requirements

- GPU: 80GB+ VRAM recommended for 67B model (or multiple GPUs with model sharding)
- CPU fallback: Available but significantly slower
- RAM: 16GB+ recommended
- Disk Space: ~130GB for model download