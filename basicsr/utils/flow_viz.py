"""Flow field visualization utilities.

Helpers to render quiver plots, heatmaps, error diagnostics, energy spectrum
curves, and comparison grids for 2D flow fields (Case1/Case2). Uses the shared
normalization helpers in ``basicsr.utils.constants`` when converting to/from
physical units in examples.
"""

from typing import Dict, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from basicsr.metrics.spectrum_metric import _compute_2d_energy_spectrum
from basicsr.utils.constants import denormalize, normalize


def _auto_subsample(u: np.ndarray, v: np.ndarray, user_subsample: int) -> int:
    """Auto-pick a subsample factor based on resolution."""
    h, w = u.shape
    auto_factor = max(1, int(max(h, w) / 60))  # target ~60 arrows on longest side
    return max(user_subsample, auto_factor)


def plot_flow_quiver(
    u: np.ndarray,
    v: np.ndarray,
    title: str = "",
    subsample: int = 1,
    scale: float = 20,
    colorbar: bool = True,
):
    """Plot a quiver (arrow) visualization of a 2D flow field.

    Args:
        u (ndarray): Horizontal velocity component (H, W) in physical units.
        v (ndarray): Vertical velocity component (H, W) in physical units.
        title (str): Figure title.
        subsample (int): Draw one arrow every N points; auto-raises for high-res grids.
        scale (float): Arrow scaling passed to ``matplotlib.pyplot.quiver``.
        colorbar (bool): Show colorbar for speed magnitude.

    Returns:
        matplotlib.figure.Figure: Figure containing the quiver plot.

    Example:
        >>> u_phys = denormalize(u_norm, mean=0.244449, std=0.266751)
        >>> v_phys = denormalize(v_norm, mean=0.244449, std=0.266751)
        >>> fig = plot_flow_quiver(u_phys, v_phys, title='Case1 Flow')
        >>> fig.savefig('quiver.png', dpi=300)
    """
    eff_subsample = _auto_subsample(u, v, subsample)
    u_sub = u[::eff_subsample, ::eff_subsample]
    v_sub = v[::eff_subsample, ::eff_subsample]
    speed = np.sqrt(u_sub**2 + v_sub**2)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    q = ax.quiver(
        u_sub,
        v_sub,
        speed,
        scale=scale,
        cmap="viridis",
        angles="xy",
        scale_units="xy",
    )
    ax.set_aspect("equal")
    ax.set_title(title or "Flow Quiver")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()  # keep array origin at top-left
    if colorbar:
        fig.colorbar(q, ax=ax, label="Speed magnitude")
    fig.tight_layout()
    return fig


def plot_flow_heatmap(
    data: np.ndarray,
    title: str = "",
    cmap: str = "RdBu_r",
    vmin=None,
    vmax=None,
):
    """Plot a heatmap for a single flow component.

    Args:
        data (ndarray): 2D array for one component (u or v) in physical units.
        title (str): Figure title.
        cmap (str): Matplotlib colormap name.
        vmin: Color minimum; auto-set symmetric around zero if None with vmax None.
        vmax: Color maximum; auto-set symmetric around zero if None with vmin None.

    Returns:
        matplotlib.figure.Figure: Figure containing the heatmap.

    Example:
        >>> data_phys = denormalize(data_norm, mean=0.991759, std=1.087655)
        >>> fig = plot_flow_heatmap(data_phys, title='U component (Case2)')
        >>> fig.savefig('u_heatmap.png', dpi=300)
    """
    if vmin is None and vmax is None:
        max_abs = np.max(np.abs(data)) if data.size else 0
        if max_abs > 0:
            norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
            vmin_use, vmax_use = None, None  # Don't pass vmin/vmax when using norm
        else:
            norm = None
            vmin_use, vmax_use = None, None
    else:
        norm = None
        vmin_use, vmax_use = vmin, vmax

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(data, cmap=cmap, vmin=vmin_use, vmax=vmax_use, norm=norm, origin="upper")
    ax.set_title(title or "Flow Heatmap")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_error_heatmap(
    gt: np.ndarray,
    preds: Sequence[np.ndarray],
    method_names: Sequence[str],
    title_prefix: str = "",
):
    """Plot error magnitude heatmaps for multiple prediction methods.

    Args:
        gt (ndarray): Ground truth velocity field, shape (H, W, 2).
        preds (Sequence[ndarray]): Predicted fields, each (H, W, 2).
        method_names (Sequence[str]): Names corresponding to ``preds``.
        title_prefix (str): Optional prefix for subplot titles.

    Returns:
        matplotlib.figure.Figure: Figure with one column per method showing |pred-gt|.

    Example:
        >>> fig = plot_error_heatmap(gt, [pred1, pred2], ['Nearest', 'Bilinear'])
        >>> fig.savefig('error_heatmap.png', dpi=300)
    """
    assert len(preds) == len(method_names), "preds and method_names must align"
    num_methods = len(preds)

    errors = [np.linalg.norm(pred - gt, axis=-1) for pred in preds]
    vmax = max(np.max(err) for err in errors)

    fig, axes = plt.subplots(
        1, num_methods, figsize=(4 * num_methods, 4), dpi=300, constrained_layout=True
    )
    if num_methods == 1:
        axes = [axes]

    ims = []
    for ax, err, name in zip(axes, errors, method_names):
        im = ax.imshow(err, cmap="inferno", vmin=0, vmax=vmax, origin="upper")
        ax.set_title(f"{title_prefix}{name}".strip())
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ims.append(im)

    cbar = fig.colorbar(ims[0], ax=axes, fraction=0.046, pad=0.04, location="right")
    cbar.set_label("Error magnitude")
    return fig


def plot_error_histogram(
    gt: np.ndarray, preds: Sequence[np.ndarray], method_names: Sequence[str]
):
    """Plot per-method error histograms with RMSE and MAE annotations.

    Args:
        gt (ndarray): Ground truth velocity field, shape (H, W, 2).
        preds (Sequence[ndarray]): Predicted fields, each (H, W, 2).
        method_names (Sequence[str]): Names corresponding to ``preds``.
    Returns:
        matplotlib.figure.Figure: Figure containing histograms for all methods.

    Example:
        >>> fig = plot_error_histogram(gt, [pred1, pred2], ['Nearest', 'Bilinear'])
        >>> fig.savefig('error_hist.png', dpi=300)
    """
    assert len(preds) == len(method_names), "preds and method_names must align"
    num_methods = len(preds)

    errors = [np.linalg.norm(pred - gt, axis=-1).ravel() for pred in preds]
    all_errors = np.concatenate(errors)
    bins = np.linspace(all_errors.min(), all_errors.max(), 50) if all_errors.size else 10

    fig, axes = plt.subplots(
        1, num_methods, figsize=(5 * num_methods, 4), dpi=300, constrained_layout=True
    )
    if num_methods == 1:
        axes = [axes]

    for ax, err, name in zip(axes, errors, method_names):
        rmse = float(np.sqrt(np.mean(err**2))) if err.size else 0.0
        mae = float(np.mean(np.abs(err))) if err.size else 0.0
        ax.hist(err, bins=bins, color="#1f77b4", alpha=0.7, density=True)
        ax.set_title(name)
        ax.set_xlabel("Error magnitude")
        ax.set_ylabel("Density")
        ax.text(
            0.97,
            0.95,
            f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    return fig


def plot_energy_spectrum(
    data_dict: Dict[str, np.ndarray],
    title: str = "Energy Spectrum Comparison",
):
    """Plot energy spectra for multiple flow fields on a shared chart.

    Args:
        data_dict (dict): Mapping of name -> velocity field (H, W, 2) in physical units.
        title (str): Figure title.

    Returns:
        matplotlib.figure.Figure: Figure containing the spectrum curves.

    Example:
        >>> data_phys = denormalize(pred_norm, mean=0.991759, std=1.087655)
        >>> fig = plot_energy_spectrum({'GT': gt_phys, 'SR': data_phys})
        >>> fig.savefig('spectrum.png', dpi=300)
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for name, field in data_dict.items():
        spectrum = _compute_2d_energy_spectrum(field)
        k = np.arange(len(spectrum))
        ax.loglog(k[1:], spectrum[1:] + 1e-12, label=name)  # skip K=0 to avoid log issues

    ax.set_xlabel("Wavenumber (K)")
    ax.set_ylabel("Energy Spectrum E(K)")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def create_comparison_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    ncols: int = 6,
    figsize: Tuple[float, float] = (24, 4),
):
    """Create a grid of flow visualizations (quiver or heatmap depending on shape).

    Args:
        images (Sequence[ndarray]): List of images. (H, W, 2) entries render as quiver;
            others render as heatmaps.
        titles (Sequence[str]): Titles for each subplot.
        ncols (int): Number of columns.
        figsize (tuple): Figure size passed to Matplotlib.

    Returns:
        matplotlib.figure.Figure: Figure containing the comparison grid.

    Example:
        >>> imgs = [gt, pred1, pred2]
        >>> titles = ['GT', 'Nearest', 'Bilinear']
        >>> fig = create_comparison_grid(imgs, titles, ncols=3)
        >>> fig.savefig('grid.png', dpi=300)
    """
    assert len(images) == len(titles), "images and titles must align"
    n_items = len(images)
    nrows = int(np.ceil(n_items / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=300, squeeze=False)

    for idx, (img, title) in enumerate(zip(images, titles)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        if img.ndim == 3 and img.shape[-1] == 2:
            u, v = img[..., 0], img[..., 1]
            eff_subsample = _auto_subsample(u, v, user_subsample=1)
            u_sub = u[::eff_subsample, ::eff_subsample]
            v_sub = v[::eff_subsample, ::eff_subsample]
            speed = np.sqrt(u_sub**2 + v_sub**2)
            q = ax.quiver(
                u_sub,
                v_sub,
                speed,
                scale=20,
                cmap="viridis",
                angles="xy",
                scale_units="xy",
            )
            fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
            ax.set_aspect("equal")
        else:
            im = ax.imshow(img, cmap="RdBu_r", origin="upper")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    for j in range(n_items, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col].axis("off")

    fig.tight_layout()
    return fig
