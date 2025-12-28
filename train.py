"""
Training script for control vectors using repeng.
"""

import argparse
import torch
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel

from dataset import (
    make_dataset_from_concept,
    PERSONA_PAIRS,
)
from layers import get_recommended_layers


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


def train_control_vector(
    model_name: str,
    concept: str,
    output_path: Optional[str] = None,
    layers: Optional[List[int]] = None,
    device: str = "auto",
    export_gguf: bool = False,
) -> ControlVector:
    """
    Train a control vector for a given concept.

    Args:
        model_name: HuggingFace model name
        concept: Concept name from PERSONA_PAIRS
        output_path: Optional path to save the vector
        layers: Optional list of layers to target
        device: Device to use ("auto", "cuda", "cpu")
        export_gguf: Whether to export as GGUF for llama.cpp

    Returns:
        Trained ControlVector
    """
    # Load model
    model, tokenizer = load_model(model_name, device)

    # Get layers
    if layers is None:
        num_layers = model.config.num_hidden_layers
        layers = get_recommended_layers(num_layers)

    print(f"Using layers: {layers[0]} to {layers[-1]}")

    # Wrap model
    control_model = ControlModel(model, layers)

    # Create dataset
    print(f"Creating dataset for concept: {concept}")
    dataset = make_dataset_from_concept(concept, model_name=model_name)
    print(f"Dataset size: {len(dataset)} entries")

    # Train vector
    print("Training control vector...")
    control_model.reset()
    vector = ControlVector.train(control_model, tokenizer, dataset)
    print("Training complete!")

    # Save if path provided
    if output_path:
        if export_gguf:
            gguf_path = output_path.replace(".pt", ".gguf")
            vector.export_gguf(gguf_path)
            print(f"Exported GGUF to: {gguf_path}")
        else:
            torch.save(vector, output_path)
            print(f"Saved vector to: {output_path}")

    return vector


def main():
    parser = argparse.ArgumentParser(description="Train a control vector")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        choices=list(PERSONA_PAIRS.keys()),
        help="Concept to train vector for"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the vector"
    )
    parser.add_argument(
        "--gguf",
        action="store_true",
        help="Export as GGUF for llama.cpp"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use"
    )

    args = parser.parse_args()

    output_path = args.output or f"{args.concept}_vector.pt"

    train_control_vector(
        model_name=args.model,
        concept=args.concept,
        output_path=output_path,
        device=args.device,
        export_gguf=args.gguf,
    )


if __name__ == "__main__":
    main()
