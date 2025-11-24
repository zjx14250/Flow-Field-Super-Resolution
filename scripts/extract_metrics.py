#!/usr/bin/env python3
"""Extract best metrics from experiment logs and emit a Markdown summary."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional


# Regular expression provided in the requirements for parsing metric lines.
METRIC_PATTERN = re.compile(r"# (\w+): ([\d.]+)\s+Best: ([\d.]+) @ (\d+) iter")

# Metric ordering along with the formatter that should be used for Markdown output.
METRIC_DEFS = [
    ("psnr", "{:.2f}"),
    ("ssim", "{:.4f}"),
    ("rmse", "{:.2f}"),
    ("pearsonr", "{:.4f}"),
    ("energy_spectrum", "{:.4f}"),
]

BASELINE_METHODS = ["Nearest", "Bilinear", "Bicubic"]
METHOD_ORDER = {name: idx for idx, name in enumerate(BASELINE_METHODS + ["SwinIR"])}
SCALES = ["x2", "x4", "x8"]
SCALE_ORDER = {scale: idx for idx, scale in enumerate(SCALES)}
DATASET_ORDER = {f"Case{case}": idx for idx, case in enumerate([1, 2])}


@dataclass
class ExperimentSpec:
    """Lightweight description of an experiment that needs to be parsed."""

    dataset: str
    method: str
    scale: str
    directory: Path


@dataclass
class ExperimentResult:
    """Holds metrics that were successfully parsed for an experiment."""

    dataset: str
    method: str
    scale: str
    metrics: Dict[str, float]


def warn(message: str) -> None:
    """Print warnings to stderr so they do not mix with Markdown output."""

    print(f"[WARN] {message}", file=sys.stderr)


def repo_root() -> Path:
    """Return the repository root inferred from the location of this script."""

    return Path(__file__).resolve().parents[1]


def experiments_root(root: Path) -> Path:
    """Return the canonical experiments directory."""

    return root / "experiments"


def build_specs(exp_root: Path) -> List[ExperimentSpec]:
    """Enumerate all Baseline and SwinIR experiment descriptors."""

    specs: List[ExperimentSpec] = []
    for case in (1, 2):
        dataset = f"Case{case}"
        # Baseline experiments include a method suffix (Nearest/Bilinear/Bicubic).
        for method in BASELINE_METHODS:
            for scale in SCALES:
                dir_name = f"Baseline_{method}_Case{case}_{scale}"
                specs.append(
                    ExperimentSpec(dataset, method, scale, exp_root / dir_name)
                )
        # SwinIR experiments only vary by case and scale.
        for scale in SCALES:
            dir_name = f"SwinIR_Case{case}_{scale}"
            specs.append(
                ExperimentSpec(dataset, "SwinIR", scale, exp_root / dir_name)
            )
    return specs


def latest_log_file(exp_dir: Path) -> Optional[Path]:
    """Return the newest train_*.log file inside the experiment directory."""

    # Collect candidates and sort by modification time so the freshest log wins.
    log_candidates = sorted(
        exp_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return log_candidates[0] if log_candidates else None


def parse_log_metrics(log_path: Path) -> Optional[Dict[str, float]]:
    """Parse metrics that appear after the last 'End of training' marker."""

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        warn(f"Failed to read log {log_path}: {exc}")
        return None

    pivot = log_text.rfind("End of training")
    if pivot == -1:
        warn(f"'End of training' marker not found in {log_path}")
        return None

    metrics: Dict[str, float] = {}
    for match in METRIC_PATTERN.finditer(log_text[pivot:]):
        metric_name = match.group(1).lower()
        try:
            best_val = float(match.group(3))
        except (TypeError, ValueError):
            warn(f"Could not parse metric value for {metric_name} in {log_path}")
            continue
        metrics[metric_name] = best_val

    required = [name for name, _ in METRIC_DEFS]
    missing = [name for name in required if name not in metrics]
    if missing:
        warn(
            f"Metrics {', '.join(missing)} missing in {log_path} (only found {sorted(metrics)})"
        )
        return None

    return metrics


def collect_results(specs: Iterable[ExperimentSpec]) -> List[ExperimentResult]:
    """Collect all experiment results while honoring the warning requirements."""

    results: List[ExperimentResult] = []
    for spec in specs:
        # Requirement: warn and skip missing experiment directories instead of failing.
        if not spec.directory.exists() or not spec.directory.is_dir():
            warn(f"Experiment directory missing: {spec.directory}")
            continue
        log_path = latest_log_file(spec.directory)
        if log_path is None:
            warn(f"No train_*.log files found in {spec.directory}")
            continue
        metrics = parse_log_metrics(log_path)
        if metrics is None:
            continue
        results.append(ExperimentResult(spec.dataset, spec.method, spec.scale, metrics))
    return results


def sort_results(results: Iterable[ExperimentResult]) -> List[ExperimentResult]:
    """Sort rows by Dataset -> Method -> Scale using the prescribed order."""

    def sort_key(res: ExperimentResult):
        return (
            DATASET_ORDER.get(res.dataset, 999),
            METHOD_ORDER.get(res.method, 999),
            SCALE_ORDER.get(res.scale, 999),
        )

    return sorted(results, key=sort_key)


def format_markdown(results: List[ExperimentResult]) -> str:
    """Return the Markdown table and notes as a single string."""

    header = (
        "| Dataset | Method | Scale | PSNR ↑ | SSIM ↑ | RMSE ↓ | Pearson ↑ | Energy Spectrum ↓ |\n"
        "|---------|--------|-------|--------|---------|---------|----------|--------------------|"
    )

    lines = [header]

    for res in results:
        formatted_metrics = [format_metric(res.metrics[name], fmt) for name, fmt in METRIC_DEFS]
        row = (
            f"| {res.dataset} | {res.method} | {res.scale} | "
            f"{' | '.join(formatted_metrics)} |"
        )
        lines.append(row)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    notes = [
        "",
        "Notes:",
        "- ↑ means higher is better (PSNR, SSIM, Pearson); ↓ means lower is better (RMSE, Energy Spectrum).",
        "- Training parameters: 18k iterations with shared batch_size/optimizer/etc as defined in the configs.",
        f"- Generated on {timestamp}.",
    ]

    return "\n".join(lines + notes)


def format_metric(value: float, fmt: str) -> str:
    """Safely format metric values with the desired precision."""

    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return "N/A"


def write_summary(markdown: str, output_path: Path) -> None:
    """Ensure the results directory exists and write the Markdown file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown + "\n", encoding="utf-8")


def main() -> int:
    root = repo_root()
    exp_root = experiments_root(root)

    if not exp_root.exists():
        warn(f"Experiments root not found: {exp_root}")
        return 1

    specs = build_specs(exp_root)
    results = collect_results(specs)
    ordered = sort_results(results)
    markdown = format_markdown(ordered)

    summary_path = exp_root / "results" / "summary.md"
    write_summary(markdown, summary_path)
    print(f"Wrote {len(ordered)} rows to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
