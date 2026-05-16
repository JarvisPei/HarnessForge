from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from agentdistill.metrics import build_benchmark_metrics


app = typer.Typer(add_completion=False)


@app.command()
def main(
    run_dirs: list[Path] = typer.Argument(..., help="Benchmark run directories to summarize."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Optional JSON output path."),
) -> None:
    report = build_repair_efficiency_report(run_dirs)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    typer.echo(text)


def build_repair_efficiency_report(run_dirs: list[Path]) -> dict[str, Any]:
    runs = [_summarize_run(run_dir) for run_dir in run_dirs]
    return {
        "runs": runs,
        "aggregate": _aggregate_runs(runs),
    }


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    metrics = _read_json(run_dir / "metrics.json")
    train_summary = _read_json(run_dir / "train_summary.json", default=[])
    dev_impact = _read_json(run_dir / "dev_impact_report.json", default=[])
    blind_impact = _read_json(run_dir / "blind_impact_report.json", default=[])
    harness_files_after = _read_json(run_dir / "harness_files_after.json", default=[])
    if not isinstance(metrics, dict) or "repair_efficiency" not in metrics:
        if not isinstance(train_summary, list):
            train_summary = []
        if not isinstance(dev_impact, list):
            dev_impact = []
        if not isinstance(blind_impact, list):
            blind_impact = []
        if not isinstance(harness_files_after, list):
            harness_files_after = []
        metrics = build_benchmark_metrics(train_summary, dev_impact, harness_files_after, blind_impact_rows=blind_impact)
    return {
        "run_dir": str(run_dir),
        "repair_efficiency": metrics.get("repair_efficiency", {}),
        "patches": metrics.get("patches", {}),
        "dev_transfer": metrics.get("dev_transfer", {}),
        "blind_transfer": metrics.get("blind_transfer", {}),
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    patch_attempts = _sum_nested(runs, "repair_efficiency", "patch_attempts")
    accepted = _sum_nested(runs, "repair_efficiency", "accepted")
    return {
        "num_runs": len(runs),
        "patch_attempts": patch_attempts,
        "accepted": accepted,
        "rejected": _sum_nested(runs, "repair_efficiency", "rejected"),
        "accepted_rate": _safe_ratio(accepted, patch_attempts),
        "inner_repair_attempts": _sum_nested(runs, "repair_efficiency", "inner_repair_attempts"),
        "scoped_inner_repair_attempts": _sum_nested(runs, "repair_efficiency", "scoped_inner_repair_attempts"),
        "out_of_scope_rejections": _sum_nested(runs, "repair_efficiency", "out_of_scope_rejections"),
        "total_patch_paths": _sum_nested(runs, "repair_efficiency", "total_patch_paths"),
        "unique_patch_paths_sum": _sum_nested(runs, "repair_efficiency", "unique_patch_paths"),
        "dev_improved": _sum_nested(runs, "repair_efficiency", "dev", "improved"),
        "dev_regressed": _sum_nested(runs, "repair_efficiency", "dev", "regressed"),
        "blind_improved": _sum_nested(runs, "repair_efficiency", "blind", "improved"),
        "blind_regressed": _sum_nested(runs, "repair_efficiency", "blind", "regressed"),
        "teacher_call_proxy": _sum_nested(runs, "repair_efficiency", "cost_proxies", "teacher_call_proxy"),
        "weak_call_proxy": _sum_nested(runs, "repair_efficiency", "cost_proxies", "weak_call_proxy"),
        "focused_repair_weak_calls_skipped": _sum_nested(
            runs,
            "repair_efficiency",
            "cost_proxies",
            "focused_repair_weak_calls_skipped",
        ),
    }


def _sum_nested(rows: list[dict[str, Any]], *keys: str) -> int:
    total = 0
    for row in rows:
        value: Any = row
        for key in keys:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if isinstance(value, int):
            total += value
    return total


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


if __name__ == "__main__":
    app()
