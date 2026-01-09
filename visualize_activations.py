"""
Visualize activation differences across layers for control vector analysis.

Usage:
    python visualize_activations.py honesty_acts_honesty.pt
    python visualize_activations.py honesty_acts_honesty.pt --output plot.png
"""

import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import warnings


def load_activations(path: str) -> dict:
    """Load activations from .pt file."""
    return torch.load(path, map_location="cpu")


def compute_layer_differences(data: dict) -> dict:
    """
    Compute difference metrics between positive and negative activations per layer.

    Returns dict with layer_idx -> metrics
    """
    if "positive" not in data or "negative" not in data:
        raise ValueError("Data must contain 'positive' and 'negative' keys (contrastive data)")

    # Collect all layers from first sample
    first_pos = data["positive"][0]
    layers = list(first_pos["activations"].keys())

    layer_metrics = {}
    nan_warning_shown = False

    for layer_idx in layers:
        pos_norms = []
        neg_norms = []
        diff_norms = []
        cosine_sims = []

        for pos_sample, neg_sample in zip(data["positive"], data["negative"]):
            pos_act = pos_sample["activations"].get(layer_idx)
            neg_act = neg_sample["activations"].get(layer_idx)

            if pos_act is None or neg_act is None:
                continue

            # Use last token position (most relevant for generation)
            pos_last = pos_act[:, -1, :].float()
            neg_last = neg_act[:, -1, :].float()

            # Skip samples with NaN values
            if torch.isnan(pos_last).any() or torch.isnan(neg_last).any():
                if not nan_warning_shown:
                    warnings.warn(
                        "Activation data contains NaN values. These samples will be skipped. "
                        "Consider re-capturing activations with CUDA stability fixes."
                    )
                    nan_warning_shown = True
                continue

            # Compute metrics
            pos_norm = pos_last.norm().item()
            neg_norm = neg_last.norm().item()

            # Skip infinite values
            if not np.isfinite(pos_norm) or not np.isfinite(neg_norm):
                continue

            pos_norms.append(pos_norm)
            neg_norms.append(neg_norm)

            diff = pos_last - neg_last
            diff_norm = diff.norm().item()
            if np.isfinite(diff_norm):
                diff_norms.append(diff_norm)

            cosine = torch.nn.functional.cosine_similarity(
                pos_last, neg_last, dim=-1
            ).item()
            if np.isfinite(cosine):
                cosine_sims.append(cosine)

        # Handle empty lists (all samples had NaN)
        layer_metrics[layer_idx] = {
            "pos_norm_mean": np.mean(pos_norms) if pos_norms else 0.0,
            "neg_norm_mean": np.mean(neg_norms) if neg_norms else 0.0,
            "diff_norm_mean": np.mean(diff_norms) if diff_norms else 0.0,
            "diff_norm_std": np.std(diff_norms) if len(diff_norms) > 1 else 0.0,
            "cosine_sim_mean": np.mean(cosine_sims) if cosine_sims else 0.0,
            "cosine_sim_std": np.std(cosine_sims) if len(cosine_sims) > 1 else 0.0,
        }

    return layer_metrics


def plot_layer_analysis(
    layer_metrics: dict,
    concept: str = "concept",
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Create visualization of layer differences.

    Plots:
    1. Difference norm per layer (higher = more different = more important)
    2. Cosine similarity per layer (lower = more different)
    """
    layers = sorted(layer_metrics.keys())

    diff_norms = [layer_metrics[l]["diff_norm_mean"] for l in layers]
    diff_stds = [layer_metrics[l]["diff_norm_std"] for l in layers]
    cosine_sims = [layer_metrics[l]["cosine_sim_mean"] for l in layers]
    cosine_stds = [layer_metrics[l]["cosine_sim_std"] for l in layers]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Layer Analysis for '{concept}' Concept", fontsize=14, fontweight='bold')

    # Plot 1: Difference norm (bar chart)
    ax1 = axes[0, 0]
    bars = ax1.bar(layers, diff_norms, color='steelblue', alpha=0.8)
    ax1.errorbar(layers, diff_norms, yerr=diff_stds, fmt='none', color='black', capsize=2)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Difference Norm")
    ax1.set_title("Activation Difference by Layer\n(Higher = More Important for Concept)")

    # Highlight top layers
    top_k = 5
    sorted_indices = np.argsort(diff_norms)[-top_k:]
    for idx in sorted_indices:
        bars[idx].set_color('coral')

    # Plot 2: Cosine similarity
    ax2 = axes[0, 1]
    ax2.bar(layers, cosine_sims, color='seagreen', alpha=0.8)
    ax2.errorbar(layers, cosine_sims, yerr=cosine_stds, fmt='none', color='black', capsize=2)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Positive vs Negative Cosine Similarity\n(Lower = More Different)")
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Normalized difference (for comparison across models)
    ax3 = axes[1, 0]
    max_diff = max(diff_norms) if max(diff_norms) > 0 else 1
    normalized_diffs = [d / max_diff for d in diff_norms]
    ax3.fill_between(layers, normalized_diffs, alpha=0.4, color='purple')
    ax3.plot(layers, normalized_diffs, color='purple', linewidth=2)
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Normalized Difference")
    ax3.set_title("Normalized Activation Difference\n(Relative Importance)")
    ax3.set_ylim(0, 1.1)

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Find most important layers
    sorted_layers = sorted(layers, key=lambda l: layer_metrics[l]["diff_norm_mean"], reverse=True)

    summary_text = f"""
    Summary for '{concept}' Concept
    ================================

    Total Layers Analyzed: {len(layers)}

    Top 5 Most Important Layers:
    """

    for i, l in enumerate(sorted_layers[:5]):
        m = layer_metrics[l]
        summary_text += f"""
      {i+1}. Layer {l}
         Diff Norm: {m['diff_norm_mean']:.4f}
         Cosine Sim: {m['cosine_sim_mean']:.4f}
    """

    summary_text += f"""
    Recommended Layer Range:
      {sorted_layers[0]} to {sorted_layers[min(4, len(sorted_layers)-1)]}
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")

    if show:
        plt.show()

    return fig


def print_layer_ranking(layer_metrics: dict, concept: str = "concept"):
    """Print a text-based ranking of layers by importance."""

    # Filter out layers with no valid data (diff_norm_mean == 0 indicates all NaN)
    valid_layers = {
        k: v for k, v in layer_metrics.items()
        if v["diff_norm_mean"] > 0 and np.isfinite(v["diff_norm_mean"])
    }

    if not valid_layers:
        print("\n" + "=" * 60)
        print(f"Layer Ranking for '{concept}' Concept")
        print("=" * 60)
        print("ERROR: No valid layer data found. All activations may contain NaN values.")
        print("Please re-capture activations with CUDA stability fixes enabled.")
        print("=" * 60)
        return

    sorted_layers = sorted(
        valid_layers.keys(),
        key=lambda l: valid_layers[l]["diff_norm_mean"],
        reverse=True
    )

    print("\n" + "=" * 60)
    print(f"Layer Ranking for '{concept}' Concept")
    print("=" * 60)
    print(f"{'Rank':<6}{'Layer':<8}{'Diff Norm':<14}{'Cosine Sim':<14}{'Importance'}")
    print("-" * 60)

    max_diff = valid_layers[sorted_layers[0]]["diff_norm_mean"]
    if max_diff == 0:
        max_diff = 1  # Avoid division by zero

    for rank, layer in enumerate(sorted_layers, 1):
        m = valid_layers[layer]
        importance = m["diff_norm_mean"] / max_diff * 100

        # Visual bar (safely handle edge cases)
        bar_len = min(20, max(0, int(importance / 5)))
        bar = "█" * bar_len

        print(f"{rank:<6}{layer:<8}{m['diff_norm_mean']:<14.4f}{m['cosine_sim_mean']:<14.4f}{bar}")

    print("-" * 60)
    print(f"\nRecommended layers for control vector: {sorted_layers[:5]}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize activation differences across layers"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to .pt file with captured activations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot image"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting, only print text summary"
    )
    parser.add_argument(
        "--concept",
        type=str,
        default=None,
        help="Concept name for labels"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading activations from: {args.input}")
    data = load_activations(args.input)

    # Get concept name
    concept = args.concept or data.get("concept", "unknown")

    # Compute metrics
    print("Computing layer differences...")
    layer_metrics = compute_layer_differences(data)

    # Print text ranking
    print_layer_ranking(layer_metrics, concept)

    # Plot if requested
    if not args.no_plot:
        try:
            plot_layer_analysis(
                layer_metrics,
                concept=concept,
                output_path=args.output,
                show=(args.output is None)
            )
        except Exception as e:
            print(f"Could not display plot: {e}")
            if args.output:
                plot_layer_analysis(
                    layer_metrics,
                    concept=concept,
                    output_path=args.output,
                    show=False
                )


if __name__ == "__main__":
    main()
