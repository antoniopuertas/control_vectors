"""
Complete example using a TINY model: Train and test an honesty control vector.

Usage:
    python example_tiny.py

Requirements:
    - Only ~500MB RAM needed
    - Uses SmolLM-135M (~270MB download)
"""

import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry

# =========================================================================
# Configuration - Using the SMALLEST viable model
# =========================================================================

MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"  # Only 270MB!
CONCEPT = "honesty"


def make_dataset() -> List[DatasetEntry]:
    """Create a minimal dataset for quick testing."""

    # SmolLM uses ChatML-like format
    def template(persona: str, suffix: str) -> str:
        return f"<|im_start|>system\nYou are a {persona} assistant.<|im_end|>\n<|im_start|>user\n{suffix}<|im_end|>\n<|im_start|>assistant\n"

    suffixes = [
        "I think",
        "The truth is",
        "Actually,",
        "To be honest,",
        "I believe",
        "In my opinion,",
        "The answer is",
        "Let me explain:",
        "What happened was",
        "I would say",
    ]

    dataset = []
    for suffix in suffixes:
        dataset.append(
            DatasetEntry(
                positive=template("honest, truthful, sincere", suffix),
                negative=template("deceptive, dishonest, misleading", suffix),
            )
        )

    return dataset


def main():
    print("=" * 60)
    print("Control Vector Example (TINY Model - 135M params)")
    print("=" * 60)

    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU")

    print(f"Model: {MODEL_NAME}")
    print(f"Concept: {CONCEPT}")

    # =========================================================================
    # Load Model
    # =========================================================================

    print("\n[1/5] Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Small model, float32 is fine
    ).to(device)

    num_layers = model.config.num_hidden_layers
    print(f"  Model loaded! Layers: {num_layers}")

    # =========================================================================
    # Wrap Model with ControlModel
    # =========================================================================

    print("\n[2/5] Wrapping model with ControlModel...")

    # Target middle-to-late layers (45% to 85%)
    start_layer = int(num_layers * 0.45)
    end_layer = int(num_layers * 0.85)
    layers = list(range(start_layer, end_layer))

    print(f"  Targeting layers {start_layer} to {end_layer-1} (of {num_layers})")

    model = ControlModel(model, layers)

    # =========================================================================
    # Create Dataset
    # =========================================================================

    print("\n[3/5] Creating training dataset...")

    dataset = make_dataset()
    print(f"  Dataset size: {len(dataset)} pairs")

    # =========================================================================
    # Train Control Vector
    # =========================================================================

    print("\n[4/5] Training control vector...")

    model.reset()
    vector = ControlVector.train(model, tokenizer, dataset)

    print("  Training complete!")

    # =========================================================================
    # Test the Vector
    # =========================================================================

    print("\n[5/5] Testing control vector...")
    print("=" * 60)

    test_prompt = "<|im_start|>user\nDid you break the vase?<|im_end|>\n<|im_start|>assistant\n"

    coefficients = [-2, 0, 2]

    input_ids = tokenizer(test_prompt, return_tensors="pt").to(device)

    for coeff in coefficients:
        if coeff == 0:
            model.reset()
            label = "BASELINE"
        elif coeff > 0:
            model.set_control(vector, coeff)
            label = f"HONEST (+{coeff})"
        else:
            model.set_control(vector, coeff)
            label = f"DECEPTIVE ({coeff})"

        with torch.no_grad():
            output = model.generate(
                **input_ids,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract just the assistant response
        if "assistant" in response.lower():
            parts = response.split("assistant")
            response = parts[-1].strip() if len(parts) > 1 else response

        print(f"\n[{label}]")
        print("-" * 40)
        print(response[:250])

    model.reset()

    # =========================================================================
    # Interactive Demo
    # =========================================================================

    print("\n" + "=" * 60)
    print("Interactive Demo")
    print("=" * 60)
    print("\nEnter prompts to test, or 'quit' to exit.")
    print("Format: [coefficient] your prompt")
    print("Example: 2.0 Why were you late?")
    print("Example: -2.0 Did you eat my lunch?")
    print("")

    while True:
        try:
            user_input = input("\n> ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                break

            if not user_input:
                continue

            # Parse coefficient if provided
            parts = user_input.split(" ", 1)
            try:
                coeff = float(parts[0])
                prompt_text = parts[1] if len(parts) > 1 else ""
            except ValueError:
                coeff = 2.0
                prompt_text = user_input

            if not prompt_text:
                print("Please enter a prompt.")
                continue

            # Format prompt
            prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

            # Generate
            model.set_control(vector, coeff)
            input_ids = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(
                    **input_ids,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract response
            if "assistant" in response.lower():
                parts = response.split("assistant")
                response = parts[-1].strip() if len(parts) > 1 else response

            print(f"\n[Coefficient: {coeff}]")
            print(response)

            model.reset()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
