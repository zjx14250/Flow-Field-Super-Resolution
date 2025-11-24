"""Energy spectrum metric for 2D flow field super-resolution.

Computes and compares kinetic energy spectra E(K) via 2D FFT.
"""

import numpy as np

from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.constants import CASE1_MEAN, CASE1_STD, CASE2_MEAN, CASE2_STD
from basicsr.utils.registry import METRIC_REGISTRY

# Global normalization constants (must match Case1Dataset and Case1SRModel)
# Computed from all 4 resolutions: cu/zhong/xi/chao
DATA_MIN = -0.492871  # Global min from all datasets
DATA_MAX = 0.731227   # Global max from all datasets
NORM_MEAN = CASE1_MEAN  # Average mean across all resolutions
NORM_STD = CASE1_STD    # Average std across all resolutions


@METRIC_REGISTRY.register()
def calculate_energy_spectrum(img, img2, crop_border=0, input_order='HWC', **kwargs):
    """Calculate energy spectrum similarity for 2D flow field (u, v).

    Computes the kinetic energy spectrum E(K) via 2D FFT and compares
    the spectral distributions between prediction and ground truth.

    Physics background:
    - For velocity field (u, v), kinetic energy = 0.5 * (u^2 + v^2)
    - E(K) = radially-averaged energy over wavenumber magnitude K
    - Lower spectral error indicates better preservation of turbulent structures

    Args:
        img (ndarray): SR velocity field [0, 255], shape (H, W, 2)
                       Channel 0: u-component, Channel 1: v-component
        img2 (ndarray): GT velocity field [0, 255], shape (H, W, 2)
        crop_border (int): Border pixels to exclude. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        **kwargs: Additional arguments (for compatibility).

    Returns:
        float: Normalized spectral error (lower is better).
               0 = perfect spectral match.
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    # Reorder to HWC format
    img = reorder_image(img, input_order=input_order).astype(np.float64)
    img2 = reorder_image(img2, input_order=input_order).astype(np.float64)

    # Crop borders if specified
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Denormalize from [0, 255] back to physical units
    # Step 1: [0, 255] -> [DATA_MIN, DATA_MAX]
    data_range = DATA_MAX - DATA_MIN
    img_phys = (img / 255.0) * data_range + DATA_MIN
    img2_phys = (img2 / 255.0) * data_range + DATA_MIN

    # Step 2: Remove normalization (mean, std)
    img_phys = img_phys * NORM_STD + NORM_MEAN
    img2_phys = img2_phys * NORM_STD + NORM_MEAN

    # Compute energy spectra
    E_pred = _compute_2d_energy_spectrum(img_phys)
    E_gt = _compute_2d_energy_spectrum(img2_phys)

    # Compare spectra using normalized L2 distance
    spectrum_diff = np.sqrt(np.mean((E_pred - E_gt) ** 2))
    spectrum_norm = np.sqrt(np.mean(E_gt ** 2))

    # Return normalized error (0 = perfect match, smaller is better)
    return spectrum_diff / (spectrum_norm + 1e-8)


def _compute_2d_energy_spectrum(velocity_field):
    """Compute 1D radially-averaged energy spectrum from 2D velocity field.

    Args:
        velocity_field (ndarray): Shape (H, W, C), where C=2 (u, v components)

    Returns:
        ndarray: 1D energy spectrum E(K) vs wavenumber K.
                 Length = sqrt(H^2 + W^2) / 2.
    """
    if len(velocity_field.shape) == 2:
        # Single channel: treat as scalar field
        H, W = velocity_field.shape
        C = 1
        u = velocity_field
        v = np.zeros_like(u)
    else:
        H, W, C = velocity_field.shape
        if C == 1:
            # Single channel
            u = velocity_field[:, :, 0]
            v = np.zeros_like(u)
        elif C == 2:
            # Two channels (u, v)
            u = velocity_field[:, :, 0]
            v = velocity_field[:, :, 1]
        else:
            raise ValueError(f"Expected 1 or 2 channels, got {C}")

    # Compute 2D FFT for each component
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)

    # Compute kinetic energy in Fourier space: E = 0.5 * (|u_hat|^2 + |v_hat|^2)
    energy_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

    # Create wavenumber grid
    kx = np.fft.fftfreq(W, d=1.0) * W  # Wavenumber in x-direction
    ky = np.fft.fftfreq(H, d=1.0) * H  # Wavenumber in y-direction
    kx_grid, ky_grid = np.meshgrid(kx, ky)

    # Compute magnitude of wavenumber: K = sqrt(kx^2 + ky^2)
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)

    # Radially average: bin by integer wavenumber
    k_max = int(np.sqrt(H**2 + W**2) / 2)
    E_k = np.zeros(k_max)
    k_counts = np.zeros(k_max)

    k_indices = k_mag.astype(int)
    for i in range(H):
        for j in range(W):
            k_idx = k_indices[i, j]
            if 0 <= k_idx < k_max:
                E_k[k_idx] += energy_2d[i, j]
                k_counts[k_idx] += 1

    # Average over shells (avoid division by zero)
    E_k = np.divide(E_k, k_counts, out=np.zeros_like(E_k), where=k_counts > 0)

    return E_k
