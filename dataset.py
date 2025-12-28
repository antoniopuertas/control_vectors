"""
Dataset creation utilities for control vector training.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from repeng import DatasetEntry


@dataclass
class PersonaPair:
    """A pair of contrasting personas for training."""
    positive: str
    negative: str


# Common persona pairs for different concepts
PERSONA_PAIRS = {
    "honesty": PersonaPair("honest, truthful, sincere", "deceptive, dishonest, lying"),
    "creativity": PersonaPair("creative, imaginative, innovative", "uncreative, predictable, conventional"),
    "confidence": PersonaPair("confident, assertive, self-assured", "uncertain, hesitant, doubtful"),
    "helpfulness": PersonaPair("helpful, supportive, eager to assist", "unhelpful, dismissive, reluctant"),
    "formality": PersonaPair("formal, professional, polished", "casual, informal, relaxed"),
    "verbosity": PersonaPair("verbose, detailed, thorough", "concise, brief, terse"),
    "enthusiasm": PersonaPair("enthusiastic, excited, energetic", "apathetic, bored, unenthusiastic"),
    "empathy": PersonaPair("empathetic, caring, compassionate", "cold, detached, unsympathetic"),
}

# Default suffixes for training
DEFAULT_SUFFIXES = [
    "I think",
    "The answer is",
    "In my opinion,",
    "To be clear,",
    "What really happened was",
    "Let me explain:",
    "The truth is",
    "I believe",
    "From my perspective,",
    "Actually,",
    "Here's what I know:",
    "I would say",
    "The situation is",
    "My understanding is",
    "Based on my experience,",
]


def get_chat_template(model_name: str) -> Tuple[str, str]:
    """Return the appropriate chat template tags for a model."""
    templates = {
        "mistral": ("[INST]", "[/INST]"),
        "llama": ("[INST]", "[/INST]"),
        "llama3": ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
                   "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"),
        "chatml": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"),
        "phi": ("<|user|>\n", "<|end|>\n<|assistant|>\n"),
    }

    model_lower = model_name.lower()
    for key, template in templates.items():
        if key in model_lower:
            return template

    # Default to Mistral format
    return ("[INST]", "[/INST]")


def make_dataset(
    template: str,
    positive_personas: List[str],
    negative_personas: List[str],
    suffixes: List[str],
    user_tag: str = "[INST]",
    asst_tag: str = "[/INST]",
) -> List[DatasetEntry]:
    """
    Create a dataset of contrastive pairs for control vector training.

    Args:
        template: Template string with {persona} placeholder
        positive_personas: List of positive persona descriptions
        negative_personas: List of negative persona descriptions
        suffixes: List of output suffixes/continuations
        user_tag: Opening tag for user message
        asst_tag: Closing tag / assistant prefix

    Returns:
        List of DatasetEntry objects for training
    """
    dataset = []

    for suffix in suffixes:
        for pos, neg in zip(positive_personas, negative_personas):
            positive_template = template.format(persona=pos)
            negative_template = template.format(persona=neg)

            dataset.append(
                DatasetEntry(
                    positive=f"{user_tag} {positive_template} {asst_tag} {suffix}",
                    negative=f"{user_tag} {negative_template} {asst_tag} {suffix}",
                )
            )

    return dataset


def make_dataset_from_concept(
    concept: str,
    suffixes: Optional[List[str]] = None,
    model_name: str = "mistral",
    templates: Optional[List[str]] = None,
) -> List[DatasetEntry]:
    """
    Create a dataset from a predefined concept name.

    Args:
        concept: One of the keys in PERSONA_PAIRS (e.g., "honesty", "creativity")
        suffixes: Optional custom suffixes, defaults to DEFAULT_SUFFIXES
        model_name: Model name for chat template detection
        templates: Optional custom templates, defaults to standard ones

    Returns:
        List of DatasetEntry objects for training
    """
    if concept not in PERSONA_PAIRS:
        available = ", ".join(PERSONA_PAIRS.keys())
        raise ValueError(f"Unknown concept '{concept}'. Available: {available}")

    persona_pair = PERSONA_PAIRS[concept]
    suffixes = suffixes or DEFAULT_SUFFIXES
    user_tag, asst_tag = get_chat_template(model_name)

    templates = templates or [
        "Pretend you're a {persona} person making statements about the world.",
        "Act as someone who is {persona} when responding.",
        "You are a {persona} individual. Speak accordingly.",
    ]

    dataset = []
    for template in templates:
        dataset.extend(
            make_dataset(
                template=template,
                positive_personas=[persona_pair.positive],
                negative_personas=[persona_pair.negative],
                suffixes=suffixes,
                user_tag=user_tag,
                asst_tag=asst_tag,
            )
        )

    return dataset


def create_truncated_suffixes(
    texts: List[str],
    tokenizer,
    min_tokens: int = 1,
    max_tokens: int = 15,
) -> List[str]:
    """
    Create truncated versions of texts for more diverse training data.

    Args:
        texts: List of full text strings
        tokenizer: HuggingFace tokenizer
        min_tokens: Minimum number of tokens to keep
        max_tokens: Maximum number of tokens to keep

    Returns:
        List of truncated text strings
    """
    truncated = []

    for text in texts:
        tokens = tokenizer.tokenize(text)
        max_len = min(len(tokens) - 1, max_tokens)

        for i in range(min_tokens, max_len):
            truncated_text = tokenizer.convert_tokens_to_string(tokens[:i])
            if truncated_text.strip():
                truncated.append(truncated_text)

    return truncated


def load_suffixes_from_file(filepath: str) -> List[str]:
    """Load suffixes from a JSON or text file."""
    import json

    if filepath.endswith(".json"):
        with open(filepath) as f:
            return json.load(f)
    else:
        with open(filepath) as f:
            return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    # Example usage
    print("Available concepts:", list(PERSONA_PAIRS.keys()))

    # Create a simple dataset
    dataset = make_dataset_from_concept("honesty", model_name="mistral")
    print(f"\nCreated dataset with {len(dataset)} entries")
    print("\nExample entry:")
    print(f"  Positive: {dataset[0].positive[:100]}...")
    print(f"  Negative: {dataset[0].negative[:100]}...")
