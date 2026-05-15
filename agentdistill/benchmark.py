from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from agentdistill.config import BenchmarkConfig, TaskConfig, load_benchmark_config
from agentdistill.diagnosis import parse_diagnosis, write_patch_artifact
from agentdistill.harness import load_system_prompt
from agentdistill.harness_snapshot import list_harness_files, snapshot_harness
from agentdistill.metrics import build_benchmark_metrics
from agentdistill.models import ChatClient, load_model_settings
from agentdistill.patches import apply_patch_bundles_atomically
from agentdistill.report import build_impact_report, evaluate_success
from agentdistill.run import run_task
from agentdistill.tools import RuntimePolicyRegistry, ToolRegistry


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c"),
    profile: str | None = typer.Option(None, "--profile", "-p"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    load_dotenv(override=True)
    cfg = load_benchmark_config(config)
    asyncio.run(run_benchmark(cfg, profile, run_id))


async def run_benchmark(cfg: BenchmarkConfig, profile: str | None, run_id: str | None = None) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = cfg.output_dir / (profile.lower() if profile else "default") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    weak = ChatClient(load_model_settings("weak", profile))
    teacher = ChatClient(load_model_settings("teacher", profile))
    teacher_system = (repo_root / "prompts/teacher_diagnosis.md").read_text().strip()

    console.print(f"[bold]Benchmark:[/bold] {cfg.name}")
    console.print(f"[bold]Run:[/bold] {output_dir}")

    snapshot_harness(repo_root, output_dir / "harness_before")
    baseline = await _run_phase(
        cfg,
        phase="baseline_heldout",
        tasks=cfg.heldout_tasks,
        weak=weak,
        teacher=teacher,
        teacher_system=teacher_system,
        output_dir=output_dir,
        apply_patches=False,
        repo_root=repo_root,
        request_teacher_diagnosis=False,
    )
    transfer_context = _build_transfer_context(cfg.heldout_tasks, baseline)

    train_summary: list[dict[str, object]] = []
    for iteration in range(1, cfg.evolve_iterations + 1):
        phase = f"evolve_train_iter_{iteration:02d}"
        train_results = await _run_phase(
            cfg,
            phase=phase,
            tasks=cfg.train_tasks,
            weak=weak,
            teacher=teacher,
            teacher_system=teacher_system,
            output_dir=output_dir,
            apply_patches=True,
            repo_root=repo_root,
            benchmark_context=transfer_context,
        )
        train_summary.extend(
            {
                "iteration": iteration,
                "task_id": task_id,
                "applied_patch_path": result.get("applied_patch_path"),
                "applied_patch_paths": result.get("applied_patch_paths", []),
                "rejected_patch_path": result.get("rejected_patch_path"),
                "rejected_patch_paths": result.get("rejected_patch_paths", []),
                "patch_status": result.get("patch_status"),
                "contract_validation": result.get("contract_validation"),
                "harness_manifest": result.get("harness_manifest"),
                "rejection_reason": result.get("rejection_reason"),
                "failure_categories": (result.get("teacher_diagnosis") or {}).get("failure_categories", []),
            }
            for task_id, result in train_results.items()
        )
        probe_phase = f"transfer_probe_iter_{iteration:02d}"
        probe_results = await _run_phase(
            cfg,
            phase=probe_phase,
            tasks=cfg.heldout_tasks,
            weak=weak,
            teacher=teacher,
            teacher_system=teacher_system,
            output_dir=output_dir,
            apply_patches=False,
            repo_root=repo_root,
            request_teacher_diagnosis=False,
        )
        transfer_context = _build_transfer_context(cfg.heldout_tasks, probe_results)

    snapshot_harness(repo_root, output_dir / "harness_after")
    after = await _run_phase(
        cfg,
        phase="after_heldout",
        tasks=cfg.heldout_tasks,
        weak=weak,
        teacher=teacher,
        teacher_system=teacher_system,
        output_dir=output_dir,
        apply_patches=False,
        repo_root=repo_root,
        request_teacher_diagnosis=False,
    )
    transfer_context = _build_transfer_context(cfg.heldout_tasks, after)

    report_rows = build_impact_report(
        baseline=baseline,
        after=after,
        tasks=cfg.heldout_tasks,
        output_path=output_dir / "impact_report.json",
    )
    harness_files_after = list_harness_files(repo_root)
    metrics = build_benchmark_metrics(train_summary, report_rows, harness_files_after)
    (output_dir / "train_summary.json").write_text(json.dumps(train_summary, indent=2, ensure_ascii=False))
    (output_dir / "harness_files_after.json").write_text(json.dumps(harness_files_after, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    improved = sum(1 for row in report_rows if row["improved"])
    regressed = sum(1 for row in report_rows if row["regressed"])
    console.print(f"[bold]Impact:[/bold] improved={improved}, regressed={regressed}")
    console.print(f"[bold]Report:[/bold] {output_dir / 'impact_report.json'}")


async def _run_phase(
    cfg: BenchmarkConfig,
    phase: str,
    tasks: list[TaskConfig],
    weak: ChatClient,
    teacher: ChatClient,
    teacher_system: str,
    output_dir: Path,
    apply_patches: bool,
    repo_root: Path,
    benchmark_context: dict[str, object] | None = None,
    request_teacher_diagnosis: bool = True,
) -> dict[str, dict[str, object]]:
    phase_dir = output_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"\n[bold]Phase:[/bold] {phase}")
    results = {}
    summary = []
    for task in tasks:
        console.print(f"[cyan]Task[/cyan] {task.id}")
        result_path = phase_dir / f"{task.id}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            console.print(f"Reusing {result_path}")
            results[task.id] = result
            continue
        tools = ToolRegistry(cfg.harness.tools_dir)
        policies = RuntimePolicyRegistry(cfg.harness.runtime_policies_dir)
        weak_system = load_system_prompt(
            cfg.harness.system_prompt_path,
            cfg.harness.skills_dir,
            cfg.harness.guidelines_dir,
            cfg.harness.validators_dir,
            cfg.harness.tools_dir,
        )
        if tools.names:
            weak_system = weak_system + "\n\n" + tools.describe()
        result = await run_task(
            task,
            weak,
            teacher,
            weak_system,
            teacher_system,
            tools,
            policies,
            benchmark_context=benchmark_context,
            request_teacher_diagnosis=request_teacher_diagnosis,
        )
        diagnosis = parse_diagnosis(str(result["teacher_diagnosis_raw"])) if request_teacher_diagnosis else None
        if diagnosis is not None:
            result["teacher_diagnosis"] = diagnosis.model_dump()
        applied_patch_paths: list[str] = []
        if apply_patches and diagnosis is not None and diagnosis.patch_bundles and diagnosis.failure_categories:
            patch_result = apply_patch_bundles_atomically(repo_root, diagnosis.patch_bundles, task, diagnosis.harness_manifest)
            result.update(patch_result)
            applied_patch_paths = list(patch_result.get("applied_patch_paths", []))
            result["applied_patch_path"] = applied_patch_paths[0] if applied_patch_paths else None
            rejected_paths = list(patch_result.get("rejected_patch_paths", []))
            result["rejected_patch_path"] = rejected_paths[0] if rejected_paths else None
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        patch_path = write_patch_artifact(phase_dir / "patches", task.id, cfg.name, diagnosis) if diagnosis is not None else None
        summary.append(
            {
                "task_id": task.id,
                "failure_categories": diagnosis.failure_categories if diagnosis is not None else [],
                "patch_type": diagnosis.patch_type if diagnosis is not None else None,
                "result_path": str(result_path),
                "patch_path": str(patch_path) if patch_path is not None else None,
                "applied_patch_path": applied_patch_paths[0] if applied_patch_paths else None,
                "applied_patch_paths": applied_patch_paths,
                "rejected_patch_path": result.get("rejected_patch_path"),
                "rejected_patch_paths": result.get("rejected_patch_paths", []),
                "patch_status": result.get("patch_status"),
                "contract_validation": result.get("contract_validation"),
                "harness_manifest": result.get("harness_manifest"),
                "rejection_reason": result.get("rejection_reason"),
            }
        )
        results[task.id] = result
    (phase_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def _build_transfer_context(tasks: list[TaskConfig], results: dict[str, dict[str, object]]) -> dict[str, object]:
    rows = []
    for task in tasks:
        result = results.get(task.id, {})
        rows.append(
            {
                "task_id": task.id,
                "expected_answer": task.expected_answer,
                "success": evaluate_success(task, result),
                "weak_answer": result.get("weak_answer"),
                "tool_call": result.get("tool_call"),
                "tool_result": result.get("tool_result"),
                "runtime_policy_results": result.get("runtime_policy_results", []),
                "before_success": result.get("before_success"),
                "after_success": result.get("after_success"),
                "before_answer": result.get("before_answer"),
                "after_answer": result.get("after_answer"),
                "before_failures": result.get("before_failures", []),
                "after_failures": result.get("after_failures", []),
            }
        )
    return {"heldout_probe": rows}


if __name__ == "__main__":
    app()
