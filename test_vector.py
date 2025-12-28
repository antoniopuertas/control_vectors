"""
Testing and validation utilities for control vectors.
"""

import torch
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result from a single test."""
    prompt: str
    coefficient: float
    output: str
    coherence_score: float
    is_coherent: bool


def check_coherence(text: str) -> Tuple[float, List[str]]:
    """
    Check text for signs of degradation/incoherence.

    Args:
        text: Generated text to check

    Returns:
        Tuple of (score 0-1, list of warning messages)
    """
    warnings = []
    score = 1.0

    words = text.split()

    # Check for word repetition
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            warnings.append(f"High word repetition (unique ratio: {unique_ratio:.2f})")
            score -= 0.3
        elif unique_ratio < 0.5:
            warnings.append(f"Moderate word repetition (unique ratio: {unique_ratio:.2f})")
            score -= 0.15

    # Check for character repetition
    for char in "abcdefghijklmnopqrstuvwxyz.!?":
        if char * 8 in text.lower():
            warnings.append(f"Repeated character: '{char}'")
            score -= 0.2
            break

    # Check for incomplete output
    if len(text) > 50 and not any(text.rstrip().endswith(p) for p in ".!?\"')"):
        warnings.append("Text may be truncated (no ending punctuation)")
        score -= 0.1

    # Check for very short output
    if len(words) < 5 and len(text) < 30:
        warnings.append("Very short output")
        score -= 0.1

    return max(0, score), warnings


def test_vector_basic(
    model,
    tokenizer,
    vector,
    test_prompts: List[str],
    coefficients: Optional[List[float]] = None,
    max_new_tokens: int = 150,
) -> Dict[str, Dict[float, TestResult]]:
    """
    Run basic A/B comparison tests on a control vector.

    Args:
        model: ControlModel instance
        tokenizer: Tokenizer
        vector: Trained ControlVector
        test_prompts: List of prompts to test
        coefficients: List of coefficients to test
        max_new_tokens: Max tokens to generate

    Returns:
        Nested dict: prompt -> coefficient -> TestResult
    """
    if coefficients is None:
        coefficients = [-2, -1, 0, 1, 2]

    results: Dict[str, Dict[float, TestResult]] = {}

    settings = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for prompt in test_prompts:
        results[prompt] = {}
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

        for coeff in coefficients:
            if coeff == 0:
                model.reset()
            else:
                model.set_control(vector, coeff)

            with torch.no_grad():
                output = model.generate(**input_ids, **settings)

            text = tokenizer.decode(output[0], skip_special_tokens=True)
            coherence_score, warnings = check_coherence(text)

            results[prompt][coeff] = TestResult(
                prompt=prompt,
                coefficient=coeff,
                output=text,
                coherence_score=coherence_score,
                is_coherent=coherence_score > 0.7,
            )

        model.reset()

    return results


def find_optimal_coefficient(
    model,
    tokenizer,
    vector,
    test_prompt: str,
    coeff_range: Tuple[float, float] = (-3, 3),
    steps: int = 13,
    max_new_tokens: int = 100,
) -> Tuple[float, float, dict]:
    """
    Find the optimal coefficient by sweeping a range.

    Args:
        model: ControlModel instance
        tokenizer: Tokenizer
        vector: ControlVector
        test_prompt: Prompt to test
        coeff_range: (min, max) coefficient range
        steps: Number of steps to test
        max_new_tokens: Max tokens to generate

    Returns:
        Tuple of (best_positive_coeff, best_negative_coeff, all_results)
    """
    import numpy as np

    coefficients = np.linspace(coeff_range[0], coeff_range[1], steps)
    results = {}

    input_ids = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    for coeff in coefficients:
        coeff = float(coeff)

        if abs(coeff) < 0.01:
            model.reset()
        else:
            model.set_control(vector, coeff)

        with torch.no_grad():
            output = model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        coherence_score, _ = check_coherence(text)

        results[coeff] = {
            "text": text,
            "coherence": coherence_score,
        }

    model.reset()

    # Find best coherent coefficients
    positive_coeffs = [(c, r) for c, r in results.items() if c > 0 and r["coherence"] > 0.7]
    negative_coeffs = [(c, r) for c, r in results.items() if c < 0 and r["coherence"] > 0.7]

    best_positive = max(positive_coeffs, key=lambda x: x[0])[0] if positive_coeffs else 1.0
    best_negative = min(negative_coeffs, key=lambda x: x[0])[0] if negative_coeffs else -1.0

    return best_positive, best_negative, results


def print_comparison(results: Dict[str, Dict[float, TestResult]]):
    """Pretty print test results."""
    for prompt, coeff_results in results.items():
        print(f"\n{'='*70}")
        print(f"Prompt: {prompt[:60]}...")
        print('='*70)

        for coeff, result in sorted(coeff_results.items()):
            status = "OK" if result.is_coherent else "WARN"
            print(f"\n[Coeff {coeff:+.1f}] ({status}, coherence: {result.coherence_score:.2f})")
            print("-" * 40)

            # Print just the response part (after the prompt)
            response = result.output
            if prompt in response:
                response = response.split(prompt)[-1]

            # Truncate for display
            if len(response) > 300:
                response = response[:300] + "..."

            print(response.strip())


def create_test_suite(concept: str) -> Dict[str, List[str]]:
    """
    Create a test suite for a given concept.

    Args:
        concept: Concept name (e.g., "honesty", "creativity")

    Returns:
        Dictionary of test categories to prompts
    """
    suites = {
        "honesty": {
            "direct": [
                "[INST] Did you break the vase? [/INST]",
                "[INST] Are you responsible for the missing money? [/INST]",
                "[INST] Did you eat the last cookie? [/INST]",
            ],
            "hypothetical": [
                "[INST] What would you say if you made a mistake at work? [/INST]",
                "[INST] How would you explain being late to an important meeting? [/INST]",
            ],
            "control": [
                "[INST] What is 2 + 2? [/INST]",
                "[INST] Explain photosynthesis briefly. [/INST]",
            ],
        },
        "creativity": {
            "direct": [
                "[INST] Write a short poem about the moon. [/INST]",
                "[INST] Describe a new invention. [/INST]",
                "[INST] Tell me a short story about a robot. [/INST]",
            ],
            "hypothetical": [
                "[INST] If you could redesign the bicycle, what would you change? [/INST]",
            ],
            "control": [
                "[INST] What is the capital of France? [/INST]",
                "[INST] List the days of the week. [/INST]",
            ],
        },
        "confidence": {
            "direct": [
                "[INST] What's the best programming language? [/INST]",
                "[INST] Should I invest in stocks? [/INST]",
            ],
            "hypothetical": [
                "[INST] How would you introduce yourself at a job interview? [/INST]",
            ],
            "control": [
                "[INST] What year did World War 2 end? [/INST]",
            ],
        },
    }

    # Default suite for unknown concepts
    default_suite = {
        "direct": [
            f"[INST] Tell me about {concept}. [/INST]",
            f"[INST] How do you feel about {concept}? [/INST]",
        ],
        "control": [
            "[INST] What is 2 + 2? [/INST]",
        ],
    }

    return suites.get(concept, default_suite)


def run_full_evaluation(
    model,
    tokenizer,
    vector,
    concept: str,
    coefficients: Optional[List[float]] = None,
) -> dict:
    """
    Run a full evaluation of a control vector.

    Args:
        model: ControlModel instance
        tokenizer: Tokenizer
        vector: ControlVector
        concept: Concept name for test suite
        coefficients: Coefficients to test

    Returns:
        Evaluation results dictionary
    """
    if coefficients is None:
        coefficients = [-2, -1, 0, 1, 2]

    test_suite = create_test_suite(concept)
    all_results = {}

    for category, prompts in test_suite.items():
        print(f"\nTesting category: {category}")
        results = test_vector_basic(model, tokenizer, vector, prompts, coefficients)
        all_results[category] = results
        print_comparison(results)

    # Summary statistics
    total_tests = 0
    coherent_tests = 0

    for category, results in all_results.items():
        for prompt, coeff_results in results.items():
            for coeff, result in coeff_results.items():
                total_tests += 1
                if result.is_coherent:
                    coherent_tests += 1

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"Total tests: {total_tests}")
    print(f"Coherent outputs: {coherent_tests} ({100*coherent_tests/total_tests:.1f}%)")

    return {
        "results": all_results,
        "total_tests": total_tests,
        "coherent_tests": coherent_tests,
        "coherence_rate": coherent_tests / total_tests,
    }


if __name__ == "__main__":
    # Example: test coherence checker
    test_texts = [
        "This is a normal, coherent sentence.",
        "word word word word word word word word word word",
        "aaaaaaaaaaaaaaaaaaaaaaaaa",
        "Short",
    ]

    for text in test_texts:
        score, warnings = check_coherence(text)
        print(f"\nText: {text[:40]}...")
        print(f"  Score: {score:.2f}")
        print(f"  Warnings: {warnings}")
