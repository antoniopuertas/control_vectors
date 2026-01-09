"""
CUDA utilities for numerical stability and device handling.

This module addresses NaN issues that can occur with certain GPU configurations
(particularly H100 and other modern GPUs with TF32 enabled by default).
"""

import torch
from typing import Optional, Literal


def configure_cuda_for_stability():
    """
    Configure CUDA settings for numerical stability.

    This disables TF32 (TensorFloat-32) which can cause NaN values
    on certain GPU architectures due to reduced precision.

    Call this at the start of any script that uses CUDA.
    """
    if torch.cuda.is_available():
        # Disable TF32 for matmul operations
        torch.backends.cuda.matmul.allow_tf32 = False
        # Disable TF32 for cuDNN operations
        torch.backends.cudnn.allow_tf32 = False
        # Use highest precision for float32 matmul
        torch.set_float32_matmul_precision('highest')


def get_device() -> str:
    """Get the best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_device_map(device: Optional[str] = None) -> Optional[str]:
    """
    Get appropriate device_map for model loading.

    IMPORTANT: We avoid 'auto' device_map as it can cause NaN issues
    on certain GPU configurations. Use 'cuda' or None instead.

    Args:
        device: Requested device ('auto', 'cuda', 'cpu', or None)

    Returns:
        Safe device_map value for from_pretrained()
    """
    if device is None or device == "auto":
        # Convert 'auto' to explicit 'cuda' or None
        if torch.cuda.is_available():
            return "cuda"
        return None
    elif device == "cuda":
        return "cuda"
    elif device == "cpu":
        return None
    else:
        # Custom device specification
        return device


def get_dtype(
    requested_dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
) -> torch.dtype:
    """
    Get appropriate dtype for model loading.

    Args:
        requested_dtype: Requested dtype or None for auto
        device: Target device

    Returns:
        Appropriate torch.dtype
    """
    if requested_dtype is not None:
        return requested_dtype

    # Default: float32 for CPU, float16 for CUDA
    if device == "cpu" or (device is None and not torch.cuda.is_available()):
        return torch.float32
    return torch.float16


def validate_tensor(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Check if a tensor contains NaN or Inf values.

    Args:
        tensor: Tensor to validate
        name: Name for error messages

    Returns:
        True if tensor is valid (no NaN/Inf)
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        print(f"WARNING: {name} contains {'NaN' if has_nan else ''}"
              f"{' and ' if has_nan and has_inf else ''}"
              f"{'Inf' if has_inf else ''} values!")
        return False
    return True


def sanitize_tensor(
    tensor: torch.Tensor,
    replace_nan: float = 0.0,
    replace_inf: Optional[float] = None,
) -> torch.Tensor:
    """
    Replace NaN and Inf values in a tensor.

    Args:
        tensor: Input tensor
        replace_nan: Value to replace NaN with
        replace_inf: Value to replace Inf with (None = use max finite value)

    Returns:
        Sanitized tensor
    """
    result = tensor.clone()

    # Replace NaN
    nan_mask = torch.isnan(result)
    if nan_mask.any():
        result[nan_mask] = replace_nan

    # Replace Inf
    inf_mask = torch.isinf(result)
    if inf_mask.any():
        if replace_inf is not None:
            result[inf_mask] = replace_inf
        else:
            # Use dtype's max finite value
            max_val = torch.finfo(result.dtype).max
            result[result == float('inf')] = max_val
            result[result == float('-inf')] = -max_val

    return result


# Auto-configure CUDA on import
configure_cuda_for_stability()
