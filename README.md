# Control Vectors with repeng

Tools for training and testing control vectors using representation engineering.

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

## Files

| File | Description |
|------|-------------|
| `dataset.py` | Dataset creation utilities |
| `layers.py` | Layer selection utilities |
| `train.py` | Training script |
| `test_vector.py` | Testing and validation |
| `example.py` | Complete example |

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

## References

- [repeng library](https://github.com/vgel/repeng)
- [Representation Engineering paper](https://arxiv.org/abs/2310.01405)
- [Blog post](https://vgel.me/posts/representation-engineering/)
