#!/usr/bin/env python3
"""
Validate "Best" metric annotations across all required experiment logs.

This script scans 24 pre-defined experiments, extracts validation metrics,
checks whether the logged "Best" value matches the true optimum, verifies
scale-dependent trends, detects resumed training, and finally writes a
comprehensive Markdown report while printing a concise console summary.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Regular expressions provided in the task description.
VALIDATION_PATTERN = re.compile(r"INFO: Validation Case\d+Val")
METRIC_PATTERN = re.compile(r"# (\w+): ([\d.]+)\s+Best: ([\d.]+) @ (\d+) iter")
END_OF_TRAINING_PATTERN = re.compile(r"End of training")

# Metric goals determine how to compute the true best value.
METRIC_RULES: Dict[str, str] = {
    "psnr": "max",
    "ssim": "max",
    "rmse": "min",
    "pearsonr": "max",
    "energy_spectrum": "min",
}

SCALE_ORDER = ["x2", "x4", "x8"]


@dataclass
class ExperimentSpec:
    """Configuration of a single experiment log."""

    label: str
    category: str
    method: str
    dataset: str
    scale: str

    @property
    def group_key(self) -> Tuple[str, str]:
        """Group key used for trend checks."""
        return self.method, self.dataset

    @property
    def scale_value(self) -> int:
        """Integer scale factor (e.g., x4 -> 4)."""
        return int(self.scale[1:])


@dataclass
class MetricSummary:
    """Container that captures per-metric validation information."""

    actual_best: Optional[float] = None
    recorded_best: Optional[float] = None
    relative_error: Optional[float] = None
    status: str = "N/A"


@dataclass
class ExperimentResult:
    """Holds all findings for an experiment."""

    spec: ExperimentSpec
    log_path: Optional[Path] = None
    validations: List[Dict[str, Dict[str, float]]] = field(default_factory=list)
    metric_summary: Dict[str, MetricSummary] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    end_of_training_count: int = 0

    def validation_count(self) -> int:
        return len(self.validations)


def generate_specs() -> List[ExperimentSpec]:
    """Build the list of 24 experiments that must be inspected."""
    specs: List[ExperimentSpec] = []
    cases = ["Case1", "Case2"]
    scales = SCALE_ORDER.copy()

    for method in ["Nearest", "Bilinear", "Bicubic"]:
        for dataset in cases:
            for scale in scales:
                label = f"Baseline_{method}_{dataset}_{scale}"
                specs.append(
                    ExperimentSpec(
                        label=label,
                        category="Baseline",
                        method=method,
                        dataset=dataset,
                        scale=scale,
                    )
                )

    for dataset in cases:
        for scale in scales:
            label = f"SwinIR_{dataset}_{scale}"
            specs.append(
                ExperimentSpec(
                    label=label,
                    category="SwinIR",
                    method="SwinIR",
                    dataset=dataset,
                    scale=scale,
                )
            )

    return specs


def locate_log_file(exp_dir: Path) -> Optional[Path]:
    """Return the newest log file within an experiment directory."""
    if not exp_dir.exists():
        return None

    candidates = sorted(exp_dir.glob("*.log"))
    if not candidates:
        return None

    # Prefer files starting with "train" to avoid unrelated logs.
    train_logs = [p for p in candidates if p.name.startswith("train")]
    search_pool = train_logs or candidates
    return max(search_pool, key=lambda p: p.stat().st_mtime)


def parse_validations(lines: List[str]) -> List[Dict[str, Dict[str, float]]]:
    """Extract validation blocks and their metrics from log lines."""
    validations: List[Dict[str, Dict[str, float]]] = []
    current_block: Optional[Dict[str, Dict[str, float]]] = None

    for line in lines:
        if VALIDATION_PATTERN.search(line):
            current_block = {"metrics": {}}
            validations.append(current_block)
            continue

        if current_block is None:
            continue

        stripped = line.strip()
        if not stripped:
            continue

        metric_match = METRIC_PATTERN.search(stripped)
        if metric_match:
            metric_name = metric_match.group(1).lower()
            try:
                current_value = float(metric_match.group(2))
                best_value = float(metric_match.group(3))
                iteration = int(metric_match.group(4))
            except ValueError:
                # Skip malformed entries but keep scanning.
                continue

            current_block["metrics"][metric_name] = {
                "current": current_value,
                "best": best_value,
                "iter": iteration,
            }
            continue

        # Non-metric line ends the current block.
        if not stripped.startswith("#"):
            current_block = None

    return validations


def compute_relative_error(actual: float, recorded: float) -> float:
    """Return the relative difference while guarding against zero division."""
    denominator = max(abs(actual), 1e-12)
    return abs(actual - recorded) / denominator


def summarize_metrics(validations: List[Dict[str, Dict[str, float]]]) -> Dict[str, MetricSummary]:
    """Build metric summaries including true best, logged best, and status."""
    summary: Dict[str, MetricSummary] = {}

    for metric, goal in METRIC_RULES.items():
        values: List[float] = []
        recorded_best: Optional[float] = None

        for block in validations:
            entry = block["metrics"].get(metric)
            if entry is None:
                continue
            values.append(entry["current"])
            recorded_best = entry["best"]

        item = MetricSummary()

        if not values:
            item.status = "指标缺失"
            summary[metric] = item
            continue

        actual_best = max(values) if goal == "max" else min(values)
        item.actual_best = actual_best

        if recorded_best is None:
            item.status = "Best缺失"
            summary[metric] = item
            continue

        item.recorded_best = recorded_best
        rel_error = compute_relative_error(actual_best, recorded_best)
        item.relative_error = rel_error
        if rel_error > 0.01:
            item.status = "Best标记错误"
        else:
            item.status = "OK"

        summary[metric] = item

    return summary


def analyze_experiment(spec: ExperimentSpec, experiments_dir: Path) -> ExperimentResult:
    """Parse one experiment log and collect all required checks."""
    result = ExperimentResult(spec=spec)
    exp_dir = experiments_dir / spec.label

    if not exp_dir.exists():
        result.issues.append("实验目录不存在")
        return result

    log_path = locate_log_file(exp_dir)
    if log_path is None:
        result.issues.append("日志文件不存在")
        return result

    result.log_path = log_path

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        result.issues.append(f"日志读取失败: {exc}")
        return result

    result.end_of_training_count = len(END_OF_TRAINING_PATTERN.findall(log_text))
    if result.end_of_training_count > 1:
        result.issues.append(f"Resume训练 (End of training {result.end_of_training_count}次)")

    lines = log_text.splitlines()
    validations = parse_validations(lines)
    result.validations = validations

    if not validations:
        result.issues.append("无验证数据")
        return result

    metric_summary = summarize_metrics(validations)
    result.metric_summary = metric_summary

    for metric, summary in metric_summary.items():
        if summary.status == "指标缺失":
            result.issues.append(f"指标提取失败: {metric}")
        elif summary.status == "Best缺失":
            result.issues.append(f"Best标记缺失: {metric}")
        elif summary.status == "Best标记错误":
            assert summary.actual_best is not None and summary.recorded_best is not None
            assert summary.relative_error is not None
            error_percent = summary.relative_error * 100.0
            result.issues.append(
                (
                    f"Best标记错误: {metric} "
                    f"(真实 {summary.actual_best:.4f}, 日志 {summary.recorded_best:.4f}, "
                    f"误差 {error_percent:.2f}%)"
                )
            )

    return result


def collect_trend_data(results: List[ExperimentResult]) -> Dict[Tuple[str, str], Dict[str, Dict[str, Optional[float]]]]:
    """Map each (method, dataset) to per-scale best metrics."""
    trend_data: Dict[Tuple[str, str], Dict[str, Dict[str, Optional[float]]]] = {}
    for result in results:
        if not result.metric_summary:
            continue
        group = trend_data.setdefault(result.spec.group_key, {})
        metrics_snapshot = {}
        for metric in ["psnr", "rmse"]:
            summary = result.metric_summary.get(metric)
            metrics_snapshot[metric] = summary.actual_best if summary else None
        group[result.spec.scale] = metrics_snapshot
    return trend_data


def evaluate_trends(results: List[ExperimentResult]) -> List[str]:
    """Check cross-scale PSNR/RMSE monotonic trends and annotate issues."""
    anomalies: List[str] = []
    trend_data = collect_trend_data(results)

    for (method, dataset), scales in trend_data.items():
        if not all(scale in scales for scale in SCALE_ORDER):
            continue

        psnr_values = [scales[scale]["psnr"] for scale in SCALE_ORDER]
        rmse_values = [scales[scale]["rmse"] for scale in SCALE_ORDER]

        psnr_issue = _check_sequence(psnr_values, decreasing=True)
        rmse_issue = _check_sequence(rmse_values, decreasing=False)

        if psnr_issue:
            message = (
                f"趋势异常: {dataset} + {method} 的 PSNR 未满足 x2>x4>x8 "
                f"(实际 {format_sequence(psnr_values)})"
            )
            anomalies.append(message)
            _apply_trend_issue(results, method, dataset, message)

        if rmse_issue:
            message = (
                f"趋势异常: {dataset} + {method} 的 RMSE 未满足 x2<x4<x8 "
                f"(实际 {format_sequence(rmse_values)})"
            )
            anomalies.append(message)
            _apply_trend_issue(results, method, dataset, message)

    return anomalies


def _check_sequence(values: List[Optional[float]], *, decreasing: bool) -> bool:
    """Return True when the monotonic expectation is violated or data missing."""
    if any(value is None for value in values):
        return True
    if decreasing:
        return not (values[0] > values[1] > values[2])
    return not (values[0] < values[1] < values[2])


def _apply_trend_issue(results: List[ExperimentResult], method: str, dataset: str, message: str) -> None:
    """Attach the same trend issue to all experiments in the affected group."""
    for result in results:
        if result.spec.method == method and result.spec.dataset == dataset:
            if message not in result.issues:
                result.issues.append(message)


def format_sequence(values: List[Optional[float]]) -> str:
    """Render a friendly representation of scale-aligned metrics."""
    formatted = []
    for scale, value in zip(SCALE_ORDER, values):
        formatted.append(f"{scale}={value:.4f}" if value is not None else f"{scale}=--")
    return ", ".join(formatted)


def write_report(
    results: List[ExperimentResult],
    trend_messages: List[str],
    report_path: Path,
    repo_root: Path,
) -> None:
    """Persist the Markdown report that consolidates all findings."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)
    missing_logs = sum(1 for r in results if r.log_path is None)
    issue_count = sum(1 for r in results if r.issues)

    lines: List[str] = []
    lines.append("# 验证日志巡检报告")
    lines.append(f"- 生成时间: {now}")
    lines.append(f"- 实验总数: {total}")
    lines.append(f"- 缺失日志: {missing_logs}")
    lines.append(f"- 存在问题的实验: {issue_count}")
    lines.append("")

    lines.append("## 实验详情")
    for result in results:
        spec = result.spec
        lines.append(f"### {spec.label}")
        lines.append(
            f"- 方法: {spec.method} | 数据集: {spec.dataset} | Scale: {spec.scale} | 类型: {spec.category}"
        )
        if result.log_path:
            try:
                rel_path = result.log_path.relative_to(repo_root)
            except ValueError:
                rel_path = result.log_path
            lines.append(f"- 日志文件: `{rel_path}`")
        else:
            lines.append("- 日志文件: 缺失")
        lines.append(f"- 验证次数: {result.validation_count()}")
        if result.end_of_training_count > 1:
            lines.append(f"- Resume训练: 是 ({result.end_of_training_count} 次 End of training)")
        else:
            lines.append("- Resume训练: 否")

        if result.issues:
            lines.append("- 问题:")
            for issue in result.issues:
                lines.append(f"  - {issue}")
        else:
            lines.append("- 问题: 无")

        lines.append("")
        lines.append("| 指标 | 真实最优 | 日志Best | 相对误差 | 判定 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for metric in METRIC_RULES:
            summary = result.metric_summary.get(metric, MetricSummary())
            actual = (
                f"{summary.actual_best:.4f}"
                if summary.actual_best is not None
                else "--"
            )
            recorded = (
                f"{summary.recorded_best:.4f}"
                if summary.recorded_best is not None
                else "--"
            )
            rel_error = (
                f"{summary.relative_error * 100:.2f}%"
                if summary.relative_error is not None
                else "--"
            )
            lines.append(
                f"| {metric} | {actual} | {recorded} | {rel_error} | {summary.status} |"
            )

        lines.append("")

    lines.append("## 趋势检测")
    if trend_messages:
        for msg in trend_messages:
            lines.append(f"- {msg}")
    else:
        lines.append("- 未发现趋势异常。")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def print_console_summary(results: List[ExperimentResult], report_path: Path) -> None:
    """Emit progress-independent summary of discovered problems."""
    problem_lines = [
        f"{res.spec.label}: {issue}"
        for res in results
        for issue in res.issues
    ]

    print("\n问题摘要:")
    if not problem_lines:
        print("  - 未发现异常")
    else:
        for line in problem_lines:
            print(f"  - {line}")

    print(f"\n报告保存至: {report_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    experiments_dir = repo_root / "experiments"
    report_path = experiments_dir / "results" / "validation_report.md"

    specs = generate_specs()
    total = len(specs)
    results: List[ExperimentResult] = []

    for idx, spec in enumerate(specs, start=1):
        print(f"[{idx:02d}/{total}] 检查 {spec.label} ...", end="")
        result = analyze_experiment(spec, experiments_dir)
        if result.issues:
            print(f" 完成 -> 发现 {len(result.issues)} 个问题")
        else:
            print(" 完成 -> 无异常")
        results.append(result)

    trend_messages = evaluate_trends(results)
    write_report(results, trend_messages, report_path, repo_root)
    print_console_summary(results, report_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("用户中断，脚本提前结束。")
