"""
Complete example using a SMALL model: Train and test an honesty control vector.

Usage:
    python example_small.py

Requirements:
    - GPU with ~4GB VRAM (or CPU with ~8GB RAM)
    - Uses TinyLlama-1.1B (~2GB model)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry

# =========================================================================
# Configuration
# =========================================================================

# Small models that work with repeng:
# - "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  (~2GB, 22 layers)
# - "Qwen/Qwen2-1.5B-Instruct"            (~3GB, 28 layers)
# - "microsoft/phi-2"                      (~5GB, 32 layers)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CONCEPT = "honesty"


def get_tinyllama_template(persona: str, suffix: str) -> str:
    """TinyLlama uses ChatML format."""
    return f"<|system|>\nYou are a {persona} assistant.</s>\n<|user|>\nComplete this: {suffix}</s>\n<|assistant|>\n"


def make_small_dataset() -> list[DatasetEntry]:
    """Create a minimal dataset for quick testing."""

    suffixes = [
        "I think",
        "The truth is",
        "Actually,",
        "To be honest,",
        "I believe",
        "In my opinion,",
        "The answer is",
        "Let me explain:",
    ]

    dataset = []
    for suffix in suffixes:
        dataset.append(
            DatasetEntry(
                positive=get_tinyllama_template("honest, truthful, sincere", suffix),
                negative=get_tinyllama_template("deceptive, dishonest, misleading", suffix),
            )
        )

    return dataset


def main():
    print("=" * 60)
    print("Control Vector Example (Small Model)")
    print("=" * 60)

    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU (this will be slower)")

    print(f"Model: {MODEL_NAME}")
    print(f"Concept: {CONCEPT}")

    # =========================================================================
    # Load Model
    # =========================================================================

    print("\n[1/5] Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load in float16 for GPU, float32 for CPU
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    print(f"  Model loaded! Layers: {model.config.num_hidden_layers}")

    # =========================================================================
    # Wrap Model with ControlModel
    # =========================================================================

    print("\n[2/5] Wrapping model with ControlModel...")

    num_layers = model.config.num_hidden_layers

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

    dataset = make_small_dataset()
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

    test_prompt = "<|user|>\nDid you break the vase?</s>\n<|assistant|>\n"

    coefficients = [-2, 0, 2]

    input_ids = tokenizer(test_prompt, return_tensors="pt").to(model.device)

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
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=1.0,
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract just the assistant response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()

        print(f"\n[{label}]")
        print("-" * 40)
        print(response[:300])

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

            # Format for TinyLlama
            prompt = f"<|user|>\n{prompt_text}</s>\n<|assistant|>\n"

            # Generate
            model.set_control(vector, coeff)
            input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **input_ids,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()

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
