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
    if not isinstance(train_summary, list) or not train_summary:
        train_summary = _load_phase_train_summary(run_dir)
    dev_impact = _read_json(run_dir / "dev_impact_report.json", default=[])
    blind_impact = _read_json(run_dir / "blind_impact_report.json", default=[])
    harness_files_after = _read_json(run_dir / "harness_files_after.json", default=[])
    if not isinstance(train_summary, list):
        train_summary = []
    if not isinstance(dev_impact, list):
        dev_impact = []
    if not isinstance(blind_impact, list):
        blind_impact = []
    if not isinstance(harness_files_after, list):
        harness_files_after = []
    if not isinstance(metrics, dict) or _metrics_need_backfill(metrics):
        metrics = build_benchmark_metrics(train_summary, dev_impact, harness_files_after, blind_impact_rows=blind_impact)
    return {
        "run_dir": str(run_dir),
        "repair_efficiency": metrics.get("repair_efficiency", {}),
        "patches": metrics.get("patches", {}),
        "dev_transfer": metrics.get("dev_transfer", {}),
        "blind_transfer": metrics.get("blind_transfer", {}),
        "runtime_effect": metrics.get("runtime_effect", {}),
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
        "accepted_runtime_artifact": _sum_nested(runs, "patches", "accepted_runtime_artifact"),
        "accepted_test_only": _sum_nested(runs, "patches", "accepted_test_only"),
        "accepted_but_no_runtime_artifact": _sum_nested(runs, "patches", "accepted_but_no_runtime_artifact"),
        "repair_successes": _count_successes(runs, "repair_efficiency", "repair_success"),
        "inner_repair_attempts": _sum_nested(runs, "repair_efficiency", "inner_repair_attempts"),
        "inner_repair_accepted": _sum_nested(runs, "repair_efficiency", "inner_repair_accepted"),
        "scoped_inner_repair_attempts": _sum_nested(runs, "repair_efficiency", "scoped_inner_repair_attempts"),
        "scoped_inner_repair_accepted": _sum_nested(runs, "repair_efficiency", "scoped_inner_repair_accepted"),
        "scoped_inner_repair_successes": _count_successes(runs, "repair_efficiency", "scoped_inner_repair_success"),
        "out_of_scope_rejections": _sum_nested(runs, "repair_efficiency", "out_of_scope_rejections"),
        "total_patch_paths": _sum_nested(runs, "repair_efficiency", "total_patch_paths"),
        "unique_patch_paths_sum": _sum_nested(runs, "repair_efficiency", "unique_patch_paths"),
        "dev_improved": _sum_nested(runs, "repair_efficiency", "dev", "improved"),
        "dev_regressed": _sum_nested(runs, "repair_efficiency", "dev", "regressed"),
        "blind_improved": _sum_nested(runs, "repair_efficiency", "blind", "improved"),
        "blind_regressed": _sum_nested(runs, "repair_efficiency", "blind", "regressed"),
        "dev_runtime_effect": _sum_nested(runs, "runtime_effect", "dev", "after_runtime_effect"),
        "blind_runtime_effect": _sum_nested(runs, "runtime_effect", "blind", "after_runtime_effect"),
        "dev_improved_with_runtime_effect": _sum_nested(
            runs,
            "runtime_effect",
            "dev",
            "improved_with_runtime_effect",
        ),
        "blind_improved_with_runtime_effect": _sum_nested(
            runs,
            "runtime_effect",
            "blind",
            "improved_with_runtime_effect",
        ),
        "teacher_call_proxy": _sum_nested(runs, "repair_efficiency", "cost_proxies", "teacher_call_proxy"),
        "weak_call_proxy": _sum_nested(runs, "repair_efficiency", "cost_proxies", "weak_call_proxy"),
        "focused_repair_weak_calls_skipped": _sum_nested(
            runs,
            "repair_efficiency",
            "cost_proxies",
            "focused_repair_weak_calls_skipped",
        ),
    }


def _metrics_need_backfill(metrics: dict[str, Any]) -> bool:
    if "repair_efficiency" not in metrics:
        return True
    patches = metrics.get("patches", {})
    if not isinstance(patches, dict) or "accepted_runtime_artifact" not in patches:
        return True
    runtime_effect = metrics.get("runtime_effect", {})
    return not isinstance(runtime_effect, dict) or "dev" not in runtime_effect or "blind" not in runtime_effect


def _sum_nested(rows: list[dict[str, Any]], *keys: str) -> int:
    total = 0
    for row in rows:
        value: Any = row
        for key in keys:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if isinstance(value, int):
            total += value
    return total


def _count_successes(rows: list[dict[str, Any]], *keys: str) -> int:
    total = 0
    for row in rows:
        value: Any = row
        for key in keys:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if value is True:
            total += 1
    return total


def _load_phase_train_summary(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase_dir in sorted(run_dir.glob("evolve_train_iter_*")):
        if not phase_dir.is_dir():
            continue
        iteration = _iteration_from_phase(phase_dir.name)
        for result_path in sorted(phase_dir.glob("*.json")):
            if result_path.name == "summary.json" or ".inner_repair_" in result_path.name:
                continue
            result = _read_json(result_path, default={})
            if not isinstance(result, dict):
                continue
            rows.append(
                {
                    "iteration": iteration,
                    "phase_kind": "focused_repair" if result.get("focused_repair") is True else "full_train",
                    "task_id": result.get("task_id") or result_path.stem,
                    "created_at": result.get("created_at"),
                    "applied_patch_paths": result.get("applied_patch_paths", []),
                    "rejected_patch_paths": result.get("rejected_patch_paths", []),
                    "patch_status": result.get("patch_status"),
                    "contract_validation": result.get("contract_validation"),
                    "harness_manifest": result.get("harness_manifest"),
                    "context_patch_feedback": result.get("context_patch_feedback"),
                    "context_transfer_feedback": result.get("context_transfer_feedback"),
                    "context_repair_scope": result.get("context_repair_scope"),
                    "inner_repair_attempts": result.get("inner_repair_attempts", []),
                    "rejection_reason": result.get("rejection_reason"),
                    "failure_categories": (result.get("teacher_diagnosis") or {}).get("failure_categories", []),
                }
            )
    return rows


def _iteration_from_phase(name: str) -> int | None:
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


if __name__ == "__main__":
    app()
