"""Flow field metrics: RMSE and Pearson correlation.

Metrics for evaluating 2-channel flow field super-resolution.
"""

import numpy as np
from scipy.stats import pearsonr

from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_rmse(img, img2, crop_border=0, input_order='HWC', **kwargs):
    """Calculate Root Mean Square Error (RMSE) for 2-channel flow field data.

    Args:
        img (ndarray): SR flow field with range [0, 255], shape (H, W, C)
        img2 (ndarray): GT flow field with range [0, 255], shape (H, W, C)
        crop_border (int): Pixels to crop from each border. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        **kwargs: Additional arguments (for compatibility).

    Returns:
        float: RMSE value (lower is better).
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    # Reorder to HWC format
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    # Convert to float64 for precision
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Crop borders if specified
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Calculate MSE
    mse = np.mean((img - img2) ** 2)

    # Calculate RMSE
    rmse = np.sqrt(mse)

    return rmse


@METRIC_REGISTRY.register()
def calculate_pearsonr(img, img2, crop_border=0, input_order='HWC', **kwargs):
    """Calculate Pearson correlation coefficient for 2-channel flow field data.

    Computes correlation separately for each channel then averages.
    For perfect reconstruction, Pearson r = 1.0.

    Args:
        img (ndarray): SR flow field with range [0, 255], shape (H, W, C)
        img2 (ndarray): GT flow field with range [0, 255], shape (H, W, C)
        crop_border (int): Pixels to crop from each border. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        **kwargs: Additional arguments (for compatibility).

    Returns:
        float: Average Pearson correlation coefficient across channels.
               Range: [-1, 1], where 1 = perfect positive correlation.
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    # Reorder to HWC format
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    # Convert to float64 for precision
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Crop borders if specified
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Calculate Pearson correlation for each channel
    correlations = []
    num_channels = img.shape[2] if len(img.shape) == 3 else 1

    if num_channels == 1:
        # Single channel case
        pred_flat = img.flatten()
        gt_flat = img2.flatten()
        corr, _ = pearsonr(pred_flat, gt_flat)
        correlations.append(corr)
    else:
        # Multi-channel case: compute per-channel correlation
        for c in range(num_channels):
            pred_flat = img[:, :, c].flatten()
            gt_flat = img2[:, :, c].flatten()
            corr, _ = pearsonr(pred_flat, gt_flat)
            correlations.append(corr)

    # Return average correlation across channels
    return float(np.mean(correlations))
