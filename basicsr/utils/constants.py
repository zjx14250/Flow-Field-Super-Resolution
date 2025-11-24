"""Centralized normalization constants and helpers for Case1/Case2 data.

This module keeps the mean/std/min/max values used during training and
validation for the two scientific super-resolution cases. Use the helpers
here to normalize inputs and to retrieve the correct parameters per case.
"""

from typing import Tuple

# Case1 normalization parameters (see basicsr/models/case1_sr_model.py)
CASE1_MEAN = 0.244449   # NORM_MEAN
CASE1_STD = 0.266751    # NORM_STD
CASE1_MIN = -0.492871   # DATA_MIN
CASE1_MAX = 0.731227    # DATA_MAX

# Case2 normalization parameters (see basicsr/models/case2_sr_model.py)
CASE2_MEAN = 0.991759   # NORM_MEAN
CASE2_STD = 1.087655    # NORM_STD
CASE2_MIN = -1.966761   # DATA_MIN
CASE2_MAX = 2.925671    # DATA_MAX


def normalize(data, mean: float, std: float):
    """Normalize data to zero mean and unit variance.

    Args:
        data: Input values (float, numpy array, or torch tensor) to normalize.
        mean (float): Dataset mean.
        std (float): Dataset standard deviation.

    Returns:
        Same type as ``data`` with normalization applied.
    """
    return (data - mean) / std


def denormalize(data, mean: float, std: float):
    """Revert normalized data back to physical units.

    Args:
        data: Normalized values (float, numpy array, or torch tensor).
        mean (float): Dataset mean.
        std (float): Dataset standard deviation.

    Returns:
        Same type as ``data`` with denormalization applied.
    """
    return data * std + mean


def get_norm_params(case: str) -> Tuple[float, float, float, float]:
    """Return normalization parameters for a given case.

    Args:
        case (str): Either ``'case1'`` or ``'case2'`` (case-insensitive).

    Returns:
        tuple: ``(mean, std, data_min, data_max)`` for the requested case.

    Raises:
        ValueError: If ``case`` is not one of the supported options.
    """
    key = case.lower()
    if key == 'case1':
        return CASE1_MEAN, CASE1_STD, CASE1_MIN, CASE1_MAX
    if key == 'case2':
        return CASE2_MEAN, CASE2_STD, CASE2_MIN, CASE2_MAX
    raise ValueError(f"Unsupported case '{case}'. Expected 'case1' or 'case2'.")
