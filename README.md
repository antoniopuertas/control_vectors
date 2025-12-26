# Control Vectors with repeng

Tools for training and testing control vectors using representation engineering.

## Overview

This repository is a toolkit for control vectors using representation engineering - a technique to steer the behavior of large language models (LLMs) without fine-tuning.

### What it does

Control vectors allow you to modify how an LLM responds by adjusting its internal representations at specific layers. Think of it like a "personality dial" for AI models.

**Example:**
- Set honesty coefficient to `+2.0` → Model becomes more truthful
- Set honesty coefficient to `-2.0` → Model becomes more deceptive

### How it works

1. **Create contrasting prompts** - e.g., "Act as an honest assistant" vs "Act as a deceptive assistant"
2. **Capture hidden states** - Record the model's internal activations for both
3. **Compute difference vector** - The difference between these activations becomes the "control vector"
4. **Apply at inference** - Add/subtract this vector to steer behavior

### Key components

- Uses the `repeng` library under the hood
- Targets Mistral-7B model (but adaptable)
- Operates on middle layers (~14-26 for a 32-layer model)

### Use cases

- Research on AI interpretability
- Studying how models represent concepts internally
- Controllable text generation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python example.py
```

This will:
1. Load Mistral-7B
2. Train an honesty control vector
3. Test it with various coefficients
4. Launch an interactive demo

## Example Scripts

Choose the example that fits your hardware:

| Script | Model | Size | Requirements |
|--------|-------|------|--------------|
| `example.py` | Mistral-7B-Instruct-v0.1 | ~14GB | GPU with 16GB+ VRAM |
| `example_small.py` | TinyLlama-1.1B-Chat-v1.0 | ~2GB | GPU with 4GB VRAM or 8GB RAM |
| `example_tiny.py` | SmolLM-135M-Instruct | ~270MB | Only ~500MB RAM needed |

### Full example (Mistral-7B)

```bash
python example.py
```
- Model: `mistralai/Mistral-7B-Instruct-v0.1`
- Layers: 32
- Best quality results, requires significant GPU memory

### Small example (TinyLlama-1.1B)

```bash
python example_small.py
```
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Layers: 22
- Good balance between quality and resource usage
- Compatible models: `Qwen/Qwen2-1.5B-Instruct`, `microsoft/phi-2`

### Tiny example (SmolLM-135M)

```bash
python example_tiny.py
```
- Model: `HuggingFaceTB/SmolLM-135M-Instruct`
- Layers: 30
- Runs on CPU, minimal resources needed
- Great for testing and development

## Files

| File | Description |
|------|-------------|
| `dataset.py` | Dataset creation utilities |
| `layers.py` | Layer selection utilities |
| `train.py` | Training script |
| `test_vector.py` | Testing and validation |
| `example.py` | Complete example (Mistral-7B) |
| `example_small.py` | Small model example (TinyLlama-1.1B) |
| `example_tiny.py` | Tiny model example (SmolLM-135M) |
| `capture_activations.py` | Capture and save model hidden states |
| `HOWTO_activations.md` | Guide for capturing activations |

## Usage

### Train a vector from command line

```bash
python train.py --concept honesty --model mistralai/Mistral-7B-Instruct-v0.1
```

### Train programmatically

```python
from dataset import make_dataset_from_concept
from layers import get_repeng_layer_spec
from repeng import ControlVector, ControlModel

# Create dataset
dataset = make_dataset_from_concept("honesty")

# Wrap model
model = ControlModel(model, get_repeng_layer_spec(32))

# Train
vector = ControlVector.train(model, tokenizer, dataset)

# Use
model.set_control(vector, coeff=2.0)
output = model.generate(...)
```

### Available concepts

- `honesty` - truthful vs deceptive
- `creativity` - imaginative vs conventional
- `confidence` - assertive vs uncertain
- `helpfulness` - supportive vs dismissive
- `formality` - professional vs casual
- `verbosity` - detailed vs brief
- `enthusiasm` - energetic vs apathetic
- `empathy` - compassionate vs cold

### Custom concepts

```python
from dataset import make_dataset, DatasetEntry

dataset = make_dataset(
    template="Act as a {persona} assistant.",
    positive_personas=["friendly, warm"],
    negative_personas=["cold, distant"],
    suffixes=["Hello!", "How can I", "Let me"],
)
```

## Testing vectors

```python
from test_vector import test_vector_basic, print_comparison

results = test_vector_basic(
    model, tokenizer, vector,
    test_prompts=["[INST] Your question [/INST]"],
    coefficients=[-2, -1, 0, 1, 2],
)
print_comparison(results)
```

## Layer selection

```python
from layers import get_recommended_layers, get_layers_for_concept

# By model size
layers = get_recommended_layers(num_layers=32)  # [14, 15, ..., 26]

# By concept type
layers = get_layers_for_concept(32, "honesty")  # [14, 15, ..., 23]
```

## Capturing activations

Save internal model hidden states for analysis:

```bash
# Single prompt
python capture_activations.py --prompt "Tell me a story" --output my_activations

# Contrastive activations for a concept
python capture_activations.py --concept honesty --output honesty_acts

# Specific layers only
python capture_activations.py --prompt "Hello" --layers "14,15,16,17"
```

See [HOWTO_activations.md](HOWTO_activations.md) for detailed usage and examples.

## References

- [repeng library](https://github.com/vgel/repeng)
- [Representation Engineering paper](https://arxiv.org/abs/2310.01405)
- [Blog post](https://vgel.me/posts/representation-engineering/)
