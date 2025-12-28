"""
Layer selection utilities for control vectors.
"""

import torch
from typing import Optional, List, Dict
import matplotlib.pyplot as plt


# Model configurations: (total_layers, recommended_start_pct, recommended_end_pct)
MODEL_LAYER_CONFIGS = {
    "mistral-7b": (32, 0.45, 0.85),
    "mistral-8x7b": (32, 0.45, 0.85),  # MoE - may not work well
    "llama-7b": (32, 0.45, 0.85),
    "llama-13b": (40, 0.50, 0.85),
    "llama-30b": (60, 0.50, 0.85),
    "llama-65b": (80, 0.50, 0.85),
    "llama-70b": (80, 0.50, 0.85),
    "llama-2-7b": (32, 0.45, 0.85),
    "llama-2-13b": (40, 0.50, 0.85),
    "llama-2-70b": (80, 0.50, 0.85),
    "llama-3-8b": (32, 0.45, 0.85),
    "llama-3-70b": (80, 0.50, 0.85),
    "phi-2": (32, 0.45, 0.80),
    "phi-3": (32, 0.45, 0.80),
    "gemma-2b": (18, 0.45, 0.85),
    "gemma-7b": (28, 0.45, 0.85),
    "qwen-7b": (32, 0.45, 0.85),
    "qwen-14b": (40, 0.50, 0.85),
    "qwen-72b": (80, 0.50, 0.85),
}


def get_recommended_layers(
    num_layers: int,
    start_pct: float = 0.45,
    end_pct: float = 0.85,
) -> List[int]:
    """
    Get recommended layers for a model with given number of layers.

    Args:
        num_layers: Total number of transformer layers
        start_pct: Starting percentage (0.0 to 1.0)
        end_pct: Ending percentage (0.0 to 1.0)

    Returns:
        List of layer indices to target
    """
    start_layer = int(num_layers * start_pct)
    end_layer = int(num_layers * end_pct)
    return list(range(start_layer, end_layer))


def get_layers_for_model(model_name: str) -> List[int]:
    """
    Get recommended layers based on model name.

    Args:
        model_name: Model name (will try to match known configurations)

    Returns:
        List of layer indices to target
    """
    model_lower = model_name.lower()

    for key, (total, start_pct, end_pct) in MODEL_LAYER_CONFIGS.items():
        if key in model_lower:
            return get_recommended_layers(total, start_pct, end_pct)

    # Default: assume 32 layers
    print(f"Warning: Unknown model '{model_name}', assuming 32 layers")
    return get_recommended_layers(32)


def get_repeng_layer_spec(num_layers: int) -> List[int]:
    """
    Get layer specification in repeng's negative indexing format.

    Args:
        num_layers: Total number of transformer layers

    Returns:
        List of negative layer indices (e.g., [-5, -6, ..., -17])
    """
    # Target approximately layers 45% to 85%
    start_from_end = int(num_layers * 0.15)  # -5 for 32 layers
    end_from_end = int(num_layers * 0.55)    # -18 for 32 layers

    return list(range(-start_from_end, -end_from_end, -1))


def compute_layer_sensitivity(
    model,
    tokenizer,
    vector,
    test_prompt: str,
    coeff: float = 2.0,
) -> Dict[int, float]:
    """
    Compute how sensitive each layer is to the control vector.

    Args:
        model: Base HuggingFace model (not wrapped)
        tokenizer: Tokenizer
        vector: Trained ControlVector
        test_prompt: Prompt to test with
        coeff: Coefficient to apply

    Returns:
        Dictionary mapping layer index to sensitivity score
    """
    from repeng import ControlModel

    input_ids = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    num_layers = model.config.num_hidden_layers

    sensitivities = {}

    # Get baseline output
    with torch.no_grad():
        baseline_output = model.generate(
            **input_ids,
            max_new_tokens=50,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        baseline_text = tokenizer.decode(baseline_output.sequences[0])

    for layer_idx in range(num_layers):
        # Create model with single layer
        control_model = ControlModel(model, [layer_idx])
        control_model.set_control(vector, coeff)

        with torch.no_grad():
            steered_output = control_model.generate(
                **input_ids,
                max_new_tokens=50,
                do_sample=False,
            )
            steered_text = tokenizer.decode(steered_output[0])

        # Simple metric: character-level difference
        diff = sum(1 for a, b in zip(baseline_text, steered_text) if a != b)
        diff += abs(len(baseline_text) - len(steered_text))

        sensitivities[layer_idx] = float(diff)
        control_model.reset()

    return sensitivities


def find_optimal_layers(
    model,
    tokenizer,
    dataset,
    test_prompt: str,
    concept_keywords: List[str],
    coeff: float = 2.0,
    plot: bool = True,
    output_path: Optional[str] = None,
) -> List[int]:
    """
    Find optimal layers by training vectors on each layer and measuring effect.

    Args:
        model: Base HuggingFace model
        tokenizer: Tokenizer
        dataset: Training dataset
        test_prompt: Prompt to test with
        concept_keywords: Keywords to count in output (for scoring)
        coeff: Coefficient to apply
        plot: Whether to plot results
        output_path: Optional path to save plot

    Returns:
        List of best-performing layer indices
    """
    from repeng import ControlVector, ControlModel

    num_layers = model.config.num_hidden_layers
    scores = []

    print(f"Sweeping {num_layers} layers...")

    for layer in range(num_layers):
        # Train vector on single layer
        control_model = ControlModel(model, [layer])
        vector = ControlVector.train(control_model, tokenizer, dataset)

        # Test with coefficient
        control_model.set_control(vector, coeff)
        input_ids = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = control_model.generate(
                **input_ids,
                max_new_tokens=100,
                do_sample=False,
            )

        text = tokenizer.decode(output[0]).lower()

        # Score based on keyword presence
        score = sum(text.count(kw.lower()) for kw in concept_keywords)
        scores.append(score)

        control_model.reset()
        print(f"  Layer {layer}: score = {score}")

    if plot:
        plt.figure(figsize=(12, 4))
        plt.bar(range(num_layers), scores)
        plt.xlabel("Layer")
        plt.ylabel("Effect Score")
        plt.title("Layer Sensitivity Sweep")

        # Mark threshold
        if max(scores) > 0:
            threshold = max(scores) * 0.7
            plt.axhline(y=threshold, color='r', linestyle='--', label='70% threshold')
            plt.legend()

        if output_path:
            plt.savefig(output_path)
            print(f"Saved plot to: {output_path}")
        else:
            plt.show()

    # Return layers above 70% of max
    if max(scores) > 0:
        threshold = max(scores) * 0.7
        best_layers = [i for i, s in enumerate(scores) if s >= threshold]
    else:
        # Fall back to defaults if no clear signal
        best_layers = get_recommended_layers(num_layers)

    return best_layers


# Layer recommendations by concept type
CONCEPT_LAYER_HINTS = {
    "personality": (0.50, 0.80),   # Abstract behavioral patterns
    "honesty": (0.45, 0.75),       # Semantic-level decisions
    "emotion": (0.40, 0.65),       # Affects content generation
    "style": (0.55, 0.85),         # Output characteristics
    "knowledge": (0.30, 0.60),     # Factual content
    "verbosity": (0.60, 0.90),     # Output length/detail
}


def get_layers_for_concept(num_layers: int, concept_type: str) -> List[int]:
    """
    Get recommended layers based on concept type.

    Args:
        num_layers: Total transformer layers
        concept_type: Type of concept (see CONCEPT_LAYER_HINTS)

    Returns:
        List of layer indices
    """
    if concept_type in CONCEPT_LAYER_HINTS:
        start_pct, end_pct = CONCEPT_LAYER_HINTS[concept_type]
    else:
        start_pct, end_pct = 0.45, 0.85

    return get_recommended_layers(num_layers, start_pct, end_pct)


if __name__ == "__main__":
    # Example usage
    print("Layer recommendations for different models:\n")

    for model_name in ["mistral-7b", "llama-13b", "llama-70b"]:
        layers = get_layers_for_model(model_name)
        print(f"{model_name}: layers {layers[0]} to {layers[-1]} ({len(layers)} layers)")

    print("\nLayer recommendations for different concepts (32-layer model):\n")

    for concept in CONCEPT_LAYER_HINTS:
        layers = get_layers_for_concept(32, concept)
        print(f"{concept}: layers {layers[0]} to {layers[-1]}")
