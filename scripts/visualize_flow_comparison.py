"""Visualize flow SR comparisons for Case1/Case2.

Generates quiver/heatmap/error/spectrum plots for LQ upsampling baselines and SwinIR
against ground truth. Run with GPU inference and save per-sample figures.

Usage:
    python scripts/visualize_flow_comparison.py --case case1 --scale 2 --device cuda:0
"""

import argparse
import os
from typing import Dict, List, Sequence

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.data.case1_dataset import Case1Dataset
from basicsr.data.case2_dataset import Case2Dataset
from basicsr.utils.constants import denormalize, get_norm_params
from basicsr.utils.flow_viz import (
    create_comparison_grid,
    plot_error_heatmap,
    plot_error_histogram,
    plot_energy_spectrum,
)
from basicsr.utils.logger import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize flow SR comparisons for Case1/Case2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case", required=True, choices=["case1", "case2"], help="Dataset case.")
    parser.add_argument("--scale", type=int, required=True, choices=[2, 4, 8], help="Upscale factor.")
    parser.add_argument(
        "--sample_ids",
        type=str,
        default="",
        help="Comma-separated sample indices in validation set (e.g., '0,5,10'); empty = all val samples.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/visualizations",
        help="Root directory for saving figures.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument("--skip_swinir", action="store_true", help="Skip SwinIR inference (baselines only).")
    return parser.parse_args()


def resolve_paths(case: str, scale: int) -> Dict[str, str]:
    scale_to_gt = {2: "zhong", 4: "xi", 8: "chao"}
    lq_path = os.path.join("datasets", f"{case}_cu_grid.h5")
    gt_path = os.path.join("datasets", f"{case}_{scale_to_gt[scale]}_grid.h5")

    # Experiment directory uses "Case1" / "Case2" (capitalized)
    case_cap = f"Case{case[-1]}"  # case1 -> Case1, case2 -> Case2
    exp_dir = os.path.join("experiments", f"SwinIR_{case_cap}_x{scale}", "models")

    # Try net_g_latest.pth first, fall back to latest numbered checkpoint
    ckpt = os.path.join(exp_dir, "net_g_latest.pth")
    if not os.path.exists(ckpt):
        # Find latest checkpoint by sorting numerically
        ckpt_files = [f for f in os.listdir(exp_dir) if f.startswith("net_g_") and f.endswith(".pth")]
        if ckpt_files:
            # Extract iteration numbers and sort
            ckpt_iters = sorted([int(f.split("_")[-1].split(".")[0]) for f in ckpt_files if f.split("_")[-1].split(".")[0].isdigit()])
            if ckpt_iters:
                ckpt = os.path.join(exp_dir, f"net_g_{ckpt_iters[-1]}.pth")

    return {"lq": lq_path, "gt": gt_path, "ckpt": ckpt}


def build_dataset(case: str, lq_path: str, gt_path: str, sample_indices: Sequence[int]):
    dataset_cls = Case1Dataset if case == "case1" else Case2Dataset
    ds = dataset_cls({"dataroot_lq": lq_path, "dataroot_gt": gt_path, "phase": "val"})
    if sample_indices:
        # Filter by indices (0-based) into the validation set
        ds.sample_keys = [ds.sample_keys[i] for i in sample_indices if i < len(ds.sample_keys)]
    return ds


def build_swinir(case: str, scale: int, ckpt_path: str, device: str):
    net_opt = {
        "type": "SwinIR_Custom",
        "upscale": scale,
        "in_chans": 2,
        "img_size": 32,
        "window_size": 8,
        "img_range": 1.0,
        "depths": [6, 6, 6, 6],
        "embed_dim": 60,
        "num_heads": [6, 6, 6, 6],
        "mlp_ratio": 2,
        "upsampler": "pixelshuffle",
        "resi_connection": "1conv",
    }
    net = build_network(net_opt).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("params_ema") or ckpt.get("params") or ckpt
    net.load_state_dict(state, strict=False)
    net.eval()
    return net


def tensor_to_phys_np(t: torch.Tensor, mean: float, std: float) -> np.ndarray:
    arr = t.detach().cpu().float().numpy().transpose(1, 2, 0)
    return denormalize(arr, mean, std)


def save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt_close = getattr(fig, "close", None)
    if plt_close:
        plt_close()
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


def main():
    args = parse_args()
    case = args.case.lower()
    paths = resolve_paths(case, args.scale)

    # Basic checks (skip checkpoint check if skip_swinir)
    for label, p in paths.items():
        if label == "ckpt" and args.skip_swinir:
            continue
        if not os.path.exists(p):
            raise FileNotFoundError(f"{label} path not found: {p}")

    device = args.device if torch.cuda.is_available() else "cpu"
    logger = get_root_logger()
    mean, std, _, _ = get_norm_params(case)

    sample_ids = [int(s) for s in args.sample_ids.split(",") if s.strip()] if args.sample_ids else []
    dataset = build_dataset(case, paths["lq"], paths["gt"], sample_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    net_g = None if args.skip_swinir else build_swinir(case, args.scale, paths["ckpt"], device)

    out_root = os.path.join(args.output_dir, f"Case{case[-1].upper()}_x{args.scale}")

    logger.info(f"Saving visualizations to: {out_root}")
    logger.info(f"Total samples: {len(loader)}")

    print(f"Starting visualization for {len(loader)} samples...")
    for idx, data in enumerate(loader):
        print(f"Processing sample {idx+1}/{len(loader)}...")
        try:
            lq = data["lq"].to(device)
            gt = data["gt"].to(device)
            sample_key = data["lq_path"][0].split(":")[-1]
            sample_dir = os.path.join(out_root, sample_key)
            os.makedirs(sample_dir, exist_ok=True)

            # Baselines
            # LQ visualization: use nearest to preserve blocky low-resolution appearance
            lq_up = F.interpolate(lq, scale_factor=args.scale, mode="nearest")
            # SR baselines: bilinear and bicubic (nearest is same as lq_up, so skip)
            bilinear = F.interpolate(lq, scale_factor=args.scale, mode="bilinear", align_corners=False)
            bicubic = F.interpolate(lq, scale_factor=args.scale, mode="bicubic", align_corners=False)

            # SwinIR inference
            swinir = None
            if net_g is not None:
                with torch.no_grad():
                    swinir = net_g(lq).squeeze(0)
            else:
                swinir = bicubic.squeeze(0)

            # Denormalize to physical units
            lq_up_np = tensor_to_phys_np(lq_up.squeeze(0), mean, std)
            gt_np = tensor_to_phys_np(gt.squeeze(0), mean, std)
            bilinear_np = tensor_to_phys_np(bilinear.squeeze(0), mean, std)
            bicubic_np = tensor_to_phys_np(bicubic.squeeze(0), mean, std)
            swinir_np = tensor_to_phys_np(swinir, mean, std) if swinir is not None else None

            # Visualization grids (5 columns: LQ + GT + 3 methods)
            print("  Creating comparison grid (quiver)...")
            images = [lq_up_np, gt_np, bilinear_np, bicubic_np, swinir_np]
            titles = [
                f"LQ (Nearest ×{args.scale})",
                "GT",
                "Bilinear",
                "Bicubic",
                "SwinIR" if not args.skip_swinir else "Bicubic",
            ]
            fig = create_comparison_grid(images, titles, ncols=5, figsize=(20, 4))
            save_fig(fig, os.path.join(sample_dir, "comparison_quiver.png"))
            print("  Saved comparison_quiver.png")

            fig_u = create_comparison_grid(
                [img[..., 0] for img in images],
                titles,
                ncols=5,
                figsize=(20, 4),
            )
            save_fig(fig_u, os.path.join(sample_dir, "comparison_heatmap_u.png"))

            fig_v = create_comparison_grid(
                [img[..., 1] for img in images],
                titles,
                ncols=5,
                figsize=(20, 4),
            )
            save_fig(fig_v, os.path.join(sample_dir, "comparison_heatmap_v.png"))

            # Error plots (exclude LQ, include 3 baselines + SwinIR)
            pred_list = [bilinear_np, bicubic_np, swinir_np]
            method_names = ["Bilinear", "Bicubic", "SwinIR"]
            fig_err = plot_error_heatmap(gt_np, pred_list, method_names, title_prefix="")
            save_fig(fig_err, os.path.join(sample_dir, "error_heatmap.png"))

            fig_hist = plot_error_histogram(gt_np, pred_list, method_names)
            save_fig(fig_hist, os.path.join(sample_dir, "error_histogram.png"))

            data_dict: Dict[str, np.ndarray] = {
                "LQ (Nearest)": lq_up_np,
                "GT": gt_np,
                "Bilinear": bilinear_np,
                "Bicubic": bicubic_np,
                "SwinIR": swinir_np,
            }
            fig_spec = plot_energy_spectrum(data_dict, title="Energy Spectrum Comparison")
            save_fig(fig_spec, os.path.join(sample_dir, "energy_spectrum.png"))

            torch.cuda.empty_cache()
            logger.info(f"Saved visualizations for {sample_key} -> {sample_dir}")
        except Exception as exc:
            logger.error(f"Failed on sample index {idx}: {exc}")
            torch.cuda.empty_cache()
            continue


if __name__ == "__main__":
    main()
