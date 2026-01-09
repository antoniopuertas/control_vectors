"""
Complete example: Train and test an honesty control vector.

Usage:
    python example.py

Requirements:
    - GPU with ~16GB VRAM (for Mistral-7B in float16)
    - Or use a smaller model / quantization
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel

from dataset import make_dataset_from_concept, PERSONA_PAIRS
from layers import get_recommended_layers, get_repeng_layer_spec
from test_vector import test_vector_basic, print_comparison, find_optimal_coefficient
from cuda_utils import configure_cuda_for_stability, get_device_map

# Configure CUDA for numerical stability (prevents NaN on H100 and similar GPUs)
configure_cuda_for_stability()


def main():
    # =========================================================================
    # Configuration
    # =========================================================================

    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
    CONCEPT = "honesty"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {DEVICE}")
    print(f"Concept: {CONCEPT}")
    print(f"Model: {MODEL_NAME}")

    # =========================================================================
    # Load Model
    # =========================================================================

    print("\n[1/5] Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token_id = 0

    # Use safe device_map (avoid 'auto' which can cause NaN on some GPUs)
    device_map = get_device_map(DEVICE)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        device_map=device_map,
    )

    # =========================================================================
    # Wrap Model with ControlModel
    # =========================================================================

    print("\n[2/5] Wrapping model with ControlModel...")

    # Get layers (using repeng's negative indexing)
    num_layers = model.config.num_hidden_layers
    layers = get_repeng_layer_spec(num_layers)
    print(f"  Targeting layers: {layers}")

    model = ControlModel(model, layers)

    # =========================================================================
    # Create Dataset
    # =========================================================================

    print("\n[3/5] Creating training dataset...")

    dataset = make_dataset_from_concept(CONCEPT, model_name=MODEL_NAME)
    print(f"  Dataset size: {len(dataset)} entries")

    # Show example
    print(f"\n  Example positive: {dataset[0].positive[:80]}...")
    print(f"  Example negative: {dataset[0].negative[:80]}...")

    # =========================================================================
    # Train Control Vector
    # =========================================================================

    print("\n[4/5] Training control vector...")

    model.reset()
    vector = ControlVector.train(model, tokenizer, dataset)

    print("  Training complete!")

    # Save the vector
    output_path = f"{CONCEPT}_vector.pt"
    torch.save(vector, output_path)
    print(f"  Saved to: {output_path}")

    # =========================================================================
    # Test the Vector
    # =========================================================================

    print("\n[5/5] Testing control vector...")

    test_prompts = [
        "[INST] Did you break the vase? [/INST]",
        "[INST] Why were you late to work today? [/INST]",
        "[INST] Did you take the last cookie? [/INST]",
    ]

    coefficients = [-2, -1, 0, 1, 2]

    results = test_vector_basic(
        model=model,
        tokenizer=tokenizer,
        vector=vector,
        test_prompts=test_prompts,
        coefficients=coefficients,
    )

    print_comparison(results)

    # =========================================================================
    # Find Optimal Coefficient
    # =========================================================================

    print("\n" + "="*70)
    print("Finding optimal coefficient...")
    print("="*70)

    best_pos, best_neg, sweep_results = find_optimal_coefficient(
        model=model,
        tokenizer=tokenizer,
        vector=vector,
        test_prompt="[INST] Did you eat the cake that was in the fridge? [/INST]",
    )

    print(f"\n  Best positive coefficient: {best_pos}")
    print(f"  Best negative coefficient: {best_neg}")

    # =========================================================================
    # Interactive Demo
    # =========================================================================

    print("\n" + "="*70)
    print("Interactive Demo")
    print("="*70)
    print("\nEnter prompts to test, or 'quit' to exit.")
    print("Format: [coefficient] your prompt")
    print("Example: 2.0 Why did you miss the meeting?")
    print("")

    while True:
        try:
            user_input = input("\n> ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                break

            # Parse coefficient if provided
            parts = user_input.split(" ", 1)
            try:
                coeff = float(parts[0])
                prompt = parts[1] if len(parts) > 1 else ""
            except ValueError:
                coeff = 2.0
                prompt = user_input

            if not prompt:
                print("Please enter a prompt.")
                continue

            # Format prompt
            if not prompt.startswith("[INST]"):
                prompt = f"[INST] {prompt} [/INST]"

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

            # Extract just the response
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()

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
