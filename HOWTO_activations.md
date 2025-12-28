# How to Capture Model Activations

This guide explains how to capture and save internal activations (hidden states) from transformer models using `capture_activations.py`.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Command Line Usage

### Capture from a single prompt

```bash
python capture_activations.py --prompt "Tell me a story" --output my_activations
```

### Capture contrastive activations for a concept

Captures both positive and negative prompt activations for comparison:

```bash
python capture_activations.py --concept honesty --output honesty_acts --max-samples 10
```

Available concepts: `honesty`, `creativity`, `confidence`, `helpfulness`, `formality`, `verbosity`, `enthusiasm`, `empathy`

### Capture only specific layers

```bash
# Specific layers
python capture_activations.py --prompt "Hello" --layers "14,15,16,17"

# Recommended layers (45%-85% of model depth)
python capture_activations.py --prompt "Hello" --layers recommended
```

### Use PyTorch hooks (lower memory usage)

```bash
python capture_activations.py --prompt "Hello" --use-hooks
```

### Specify a different model

```bash
python capture_activations.py --model meta-llama/Llama-2-7b-hf --prompt "Hello"
```

## Output Files

| File | Contents |
|------|----------|
| `*.pt` | Tensor data (activations, input_ids, tokens) |
| `*.json` | Metadata and statistics (mean, std, norm per layer) |

## Programmatic Usage

### Basic capture

```python
from capture_activations import capture_activations, load_model

# Load model
model, tokenizer = load_model("mistralai/Mistral-7B-Instruct-v0.1")

# Capture activations
data = capture_activations(model, tokenizer, "Your prompt here")

# Access activations per layer
for layer_idx, hidden_states in data["activations"].items():
    print(f"Layer {layer_idx}: {hidden_states.shape}")
    # Shape: (batch_size, sequence_length, hidden_dim)
```

### Capture specific layers only

```python
data = capture_activations(
    model,
    tokenizer,
    "Your prompt here",
    layers=[14, 15, 16, 17, 18]
)
```

### Capture contrastive activations

```python
from capture_activations import capture_contrastive_activations

positive_prompts = [
    "Act as an honest assistant. Hello!",
    "Act as an honest assistant. How can I help?",
]
negative_prompts = [
    "Act as a deceptive assistant. Hello!",
    "Act as a deceptive assistant. How can I help?",
]

data = capture_contrastive_activations(
    model, tokenizer,
    positive_prompts,
    negative_prompts,
    layers=[14, 15, 16]
)

# Access results
for i, (pos, neg) in enumerate(zip(data["positive"], data["negative"])):
    print(f"Pair {i}: pos shape = {pos['activations'][14].shape}")
```

### Capture from predefined concept

```python
from capture_activations import capture_from_concept

data = capture_from_concept(
    model, tokenizer,
    concept="honesty",
    layers=[14, 15, 16],
    max_samples=5
)
```

### Compute statistics

```python
from capture_activations import compute_activation_stats

stats = compute_activation_stats(data)
for layer_idx, layer_stats in stats.items():
    print(f"Layer {layer_idx}:")
    print(f"  Mean: {layer_stats['mean']:.4f}")
    print(f"  Std:  {layer_stats['std']:.4f}")
    print(f"  Norm: {layer_stats['norm']:.4f}")
```

### Compute contrastive difference

```python
from capture_activations import compute_contrastive_diff

pos_data = capture_activations(model, tokenizer, "Act as honest. Hello!")
neg_data = capture_activations(model, tokenizer, "Act as deceptive. Hello!")

diffs = compute_contrastive_diff(pos_data, neg_data)
for layer_idx, diff_info in diffs.items():
    print(f"Layer {layer_idx}:")
    print(f"  Diff norm: {diff_info['diff_norm']:.4f}")
    print(f"  Cosine similarity: {diff_info['cosine_sim']:.4f}")
```

### Save and load activations

```python
from capture_activations import save_activations
import torch

# Save
save_activations(data, "my_activations")
# Creates: my_activations.pt and my_activations.json

# Load
loaded_data = torch.load("my_activations.pt")
```

## Data Structure

The captured data dictionary contains:

```python
{
    "activations": {
        0: tensor(...),   # Layer 0 hidden states
        1: tensor(...),   # Layer 1 hidden states
        # ... more layers
    },
    "input_ids": tensor(...),      # Tokenized input
    "tokens": ["<s>", "Hello", ...],  # Token strings
    "prompt": "Original prompt",
    "num_layers": 32,
    "captured_layers": [0, 1, 2, ...]
}
```

Each activation tensor has shape: `(batch_size, sequence_length, hidden_dim)`

## Tips

- **Memory**: Use `--layers` to capture only specific layers if running out of memory
- **Speed**: Use `--use-hooks` for slightly faster capture with less memory overhead
- **Analysis**: The last token position (`[:, -1, :]`) is typically most relevant for control vectors
- **Comparison**: Use `capture_contrastive_activations` to compare positive/negative prompt pairs
