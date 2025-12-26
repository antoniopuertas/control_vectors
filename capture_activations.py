"""
Capture and save model internal activations (hidden states).

This script extracts hidden states from transformer layers for analysis,
comparison, and control vector research.
"""

import argparse
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import make_dataset_from_concept, PERSONA_PAIRS
from layers import get_recommended_layers


def capture_activations(
    model,
    tokenizer,
    prompt: str,
    layers: Optional[list[int]] = None,
    use_hooks: bool = False,
) -> dict:
    """
    Capture hidden states from model layers.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        prompt: Input prompt
        layers: Specific layers to capture (None = all layers)
        use_hooks: Use PyTorch hooks instead of output_hidden_states

    Returns:
        Dictionary with activations and metadata
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    num_layers = model.config.num_hidden_layers

    if use_hooks:
        # Method 1: PyTorch hooks (more control, less memory)
        activations = {}
        hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                if layers is None or layer_idx in layers:
                    # output is a tuple, first element is hidden states
                    hidden = output[0] if isinstance(output, tuple) else output
                    activations[layer_idx] = hidden.detach().cpu()
            return hook

        # Register hooks on each layer
        for i, layer in enumerate(model.model.layers):
            h = layer.register_forward_hook(make_hook(i))
            hooks.append(h)

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Remove hooks
        for h in hooks:
            h.remove()

    else:
        # Method 2: HuggingFace output_hidden_states (simpler)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states is tuple: (embedding_output, layer_0, layer_1, ..., layer_n)
        all_hidden_states = outputs.hidden_states

        activations = {}
        for i, hidden_state in enumerate(all_hidden_states[1:]):  # Skip embedding
            if layers is None or i in layers:
                activations[i] = hidden_state.detach().cpu()

    return {
        "activations": activations,
        "input_ids": inputs["input_ids"].cpu(),
        "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
        "prompt": prompt,
        "num_layers": num_layers,
        "captured_layers": list(activations.keys()),
    }


def capture_contrastive_activations(
    model,
    tokenizer,
    positive_prompts: list[str],
    negative_prompts: list[str],
    layers: Optional[list[int]] = None,
) -> dict:
    """
    Capture activations for contrastive prompt pairs.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        positive_prompts: List of positive prompts
        negative_prompts: List of negative prompts
        layers: Specific layers to capture

    Returns:
        Dictionary with positive and negative activations
    """
    print(f"Capturing activations for {len(positive_prompts)} prompt pairs...")

    positive_activations = []
    negative_activations = []

    for i, (pos, neg) in enumerate(zip(positive_prompts, negative_prompts)):
        print(f"  Pair {i+1}/{len(positive_prompts)}")

        pos_result = capture_activations(model, tokenizer, pos, layers)
        neg_result = capture_activations(model, tokenizer, neg, layers)

        positive_activations.append(pos_result)
        negative_activations.append(neg_result)

    return {
        "positive": positive_activations,
        "negative": negative_activations,
        "num_pairs": len(positive_prompts),
        "layers": layers or list(range(model.config.num_hidden_layers)),
    }


def capture_from_concept(
    model,
    tokenizer,
    concept: str,
    layers: Optional[list[int]] = None,
    max_samples: int = 10,
) -> dict:
    """
    Capture activations using a predefined concept dataset.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        concept: Concept name from PERSONA_PAIRS
        layers: Specific layers to capture
        max_samples: Maximum number of samples to capture

    Returns:
        Dictionary with contrastive activations
    """
    dataset = make_dataset_from_concept(concept)

    positive_prompts = [entry.positive for entry in dataset[:max_samples]]
    negative_prompts = [entry.negative for entry in dataset[:max_samples]]

    result = capture_contrastive_activations(
        model, tokenizer, positive_prompts, negative_prompts, layers
    )
    result["concept"] = concept

    return result


def compute_activation_stats(activations: dict) -> dict:
    """
    Compute statistics on captured activations.

    Args:
        activations: Dictionary from capture_activations

    Returns:
        Dictionary with statistics per layer
    """
    stats = {}

    for layer_idx, tensor in activations["activations"].items():
        # tensor shape: (batch, seq_len, hidden_dim)
        stats[layer_idx] = {
            "shape": list(tensor.shape),
            "mean": tensor.mean().item(),
            "std": tensor.std().item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "norm": tensor.norm().item(),
        }

    return stats


def compute_contrastive_diff(
    positive_acts: dict,
    negative_acts: dict,
) -> dict:
    """
    Compute difference between positive and negative activations.

    Args:
        positive_acts: Activations from positive prompt
        negative_acts: Activations from negative prompt

    Returns:
        Dictionary with difference vectors per layer
    """
    diffs = {}

    for layer_idx in positive_acts["activations"]:
        if layer_idx in negative_acts["activations"]:
            pos = positive_acts["activations"][layer_idx]
            neg = negative_acts["activations"][layer_idx]

            # Use last token position (most common for control vectors)
            pos_last = pos[:, -1, :]
            neg_last = neg[:, -1, :]

            diff = pos_last - neg_last
            diffs[layer_idx] = {
                "diff_vector": diff,
                "diff_norm": diff.norm().item(),
                "cosine_sim": torch.nn.functional.cosine_similarity(
                    pos_last, neg_last, dim=-1
                ).item(),
            }

    return diffs


def save_activations(
    data: dict,
    output_path: str,
    save_tensors: bool = True,
    save_metadata: bool = True,
):
    """
    Save activations to disk.

    Args:
        data: Activation data dictionary
        output_path: Base path for output files
        save_tensors: Whether to save tensor data (.pt file)
        save_metadata: Whether to save metadata (.json file)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_tensors:
        tensor_path = output_path.with_suffix(".pt")
        torch.save(data, tensor_path)
        print(f"Saved tensors to: {tensor_path}")

    if save_metadata:
        # Check if this is contrastive data or single-prompt data
        if "positive" in data and "negative" in data:
            # Contrastive data structure
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "type": "contrastive",
                "concept": data.get("concept"),
                "num_pairs": data.get("num_pairs"),
                "layers": data.get("layers"),
                "positive_samples": [],
                "negative_samples": [],
            }

            # Add metadata for each sample
            for i, (pos, neg) in enumerate(zip(data["positive"], data["negative"])):
                pos_info = {
                    "index": i,
                    "prompt": pos.get("prompt"),
                    "num_layers": pos.get("num_layers"),
                    "captured_layers": pos.get("captured_layers"),
                    "tokens": pos.get("tokens"),
                }
                neg_info = {
                    "index": i,
                    "prompt": neg.get("prompt"),
                    "num_layers": neg.get("num_layers"),
                    "captured_layers": neg.get("captured_layers"),
                    "tokens": neg.get("tokens"),
                }

                # Add stats if activations present
                if "activations" in pos:
                    pos_info["stats"] = compute_activation_stats(pos)
                if "activations" in neg:
                    neg_info["stats"] = compute_activation_stats(neg)

                metadata["positive_samples"].append(pos_info)
                metadata["negative_samples"].append(neg_info)
        else:
            # Single-prompt data structure
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "type": "single",
                "num_layers": data.get("num_layers"),
                "captured_layers": data.get("captured_layers"),
                "prompt": data.get("prompt"),
                "tokens": data.get("tokens"),
            }

            # Add stats if activations present
            if "activations" in data:
                metadata["stats"] = compute_activation_stats(data)

        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {json_path}")


def load_model(model_name: str, device: str = "auto"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Capture model internal activations"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to capture activations for"
    )
    parser.add_argument(
        "--concept",
        type=str,
        choices=list(PERSONA_PAIRS.keys()),
        default=None,
        help="Concept for contrastive activation capture"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (e.g., '14,15,16') or 'recommended'"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="activations",
        help="Output path (without extension)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum samples for concept capture"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use"
    )
    parser.add_argument(
        "--use-hooks",
        action="store_true",
        help="Use PyTorch hooks instead of output_hidden_states"
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model, args.device)

    # Parse layers
    layers = None
    if args.layers:
        if args.layers == "recommended":
            layers = get_recommended_layers(model.config.num_hidden_layers)
        else:
            layers = [int(x.strip()) for x in args.layers.split(",")]
        print(f"Capturing layers: {layers}")

    # Capture activations
    if args.concept:
        print(f"\nCapturing contrastive activations for concept: {args.concept}")
        data = capture_from_concept(
            model, tokenizer, args.concept, layers, args.max_samples
        )
        output_path = f"{args.output}_{args.concept}"

    elif args.prompt:
        print(f"\nCapturing activations for prompt: {args.prompt[:50]}...")
        data = capture_activations(
            model, tokenizer, args.prompt, layers, args.use_hooks
        )
        output_path = args.output

    else:
        # Default: capture a simple test prompt (TinyLlama ChatML format)
        test_prompt = "<|user|>\nTell me about yourself.</s>\n<|assistant|>\n"
        print(f"\nCapturing activations for test prompt...")
        data = capture_activations(
            model, tokenizer, test_prompt, layers, args.use_hooks
        )
        output_path = args.output

    # Save
    save_activations(data, output_path)

    # Print summary
    print("\n" + "="*50)
    print("Capture complete!")
    if "activations" in data:
        print(f"Layers captured: {len(data['activations'])}")
        stats = compute_activation_stats(data)
        for layer_idx, layer_stats in list(stats.items())[:3]:
            print(f"  Layer {layer_idx}: shape={layer_stats['shape']}, "
                  f"mean={layer_stats['mean']:.4f}, std={layer_stats['std']:.4f}")
        if len(stats) > 3:
            print(f"  ... and {len(stats) - 3} more layers")


if __name__ == "__main__":
    main()
