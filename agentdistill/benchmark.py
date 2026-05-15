from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from agentdistill.config import BenchmarkConfig, TaskConfig, load_benchmark_config
from agentdistill.critic import request_policy_audit_cases
from agentdistill.diagnosis import parse_diagnosis, write_patch_artifact
from agentdistill.feedback import build_patch_feedback, merge_benchmark_context
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
    critic = ChatClient(load_model_settings("critic", profile)) if _critic_enabled(cfg.critic_mode) else None
    teacher_system = (repo_root / "prompts/teacher_diagnosis.md").read_text().strip()
    critic_system = (repo_root / "prompts/critic_policy_audit.md").read_text().strip()

    console.print(f"[bold]Benchmark:[/bold] {cfg.name}")
    console.print(f"[bold]Run:[/bold] {output_dir}")
    console.print(f"[bold]Critic mode:[/bold] {cfg.critic_mode}")

    snapshot_harness(repo_root, output_dir / "harness_before")
    dev_baseline = await _run_phase(
        cfg,
        phase="baseline_dev_probe",
        tasks=cfg.dev_probe_tasks,
        weak=weak,
        teacher=teacher,
        critic=critic,
        teacher_system=teacher_system,
        critic_system=critic_system,
        output_dir=output_dir,
        apply_patches=False,
        repo_root=repo_root,
        request_teacher_diagnosis=False,
    )
    blind_baseline = await _run_phase(
        cfg,
        phase="baseline_blind_test",
        tasks=cfg.blind_test_tasks,
        weak=weak,
        teacher=teacher,
        critic=critic,
        teacher_system=teacher_system,
        critic_system=critic_system,
        output_dir=output_dir,
        apply_patches=False,
        repo_root=repo_root,
        request_teacher_diagnosis=False,
    )
    transfer_context = _build_transfer_context(cfg.dev_probe_tasks, dev_baseline)
    patch_feedback: dict[str, object] | None = None

    train_summary: list[dict[str, object]] = []
    for iteration in range(1, cfg.evolve_iterations + 1):
        phase = f"evolve_train_iter_{iteration:02d}"
        train_results = await _run_phase(
            cfg,
            phase=phase,
            tasks=cfg.train_tasks,
            weak=weak,
            teacher=teacher,
            critic=critic,
            teacher_system=teacher_system,
            critic_system=critic_system,
            output_dir=output_dir,
            apply_patches=True,
            repo_root=repo_root,
            benchmark_context=merge_benchmark_context(transfer_context, patch_feedback),
        )
        patch_feedback = build_patch_feedback(train_results, iteration)
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
                "critic_audits": result.get("critic_audits"),
                "context_patch_feedback": result.get("context_patch_feedback"),
                "patch_feedback": patch_feedback if result.get("patch_status") == "rejected" else None,
                "rejection_reason": result.get("rejection_reason"),
                "failure_categories": (result.get("teacher_diagnosis") or {}).get("failure_categories", []),
            }
            for task_id, result in train_results.items()
        )
        probe_phase = f"transfer_probe_iter_{iteration:02d}"
        probe_results = await _run_phase(
            cfg,
            phase=probe_phase,
            tasks=cfg.dev_probe_tasks,
            weak=weak,
            teacher=teacher,
            critic=critic,
            teacher_system=teacher_system,
            critic_system=critic_system,
            output_dir=output_dir,
            apply_patches=False,
            repo_root=repo_root,
            request_teacher_diagnosis=False,
        )
        transfer_context = _build_transfer_context(cfg.dev_probe_tasks, probe_results)

    snapshot_harness(repo_root, output_dir / "harness_after")
    dev_after = await _run_phase(
        cfg,
        phase="after_dev_probe",
        tasks=cfg.dev_probe_tasks,
        weak=weak,
        teacher=teacher,
        critic=critic,
        teacher_system=teacher_system,
        critic_system=critic_system,
        output_dir=output_dir,
        apply_patches=False,
        repo_root=repo_root,
        request_teacher_diagnosis=False,
    )
    blind_after = await _run_phase(
        cfg,
        phase="after_blind_test",
        tasks=cfg.blind_test_tasks,
        weak=weak,
        teacher=teacher,
        critic=critic,
        teacher_system=teacher_system,
        critic_system=critic_system,
        output_dir=output_dir,
        apply_patches=False,
        repo_root=repo_root,
        request_teacher_diagnosis=False,
    )

    dev_report_rows = build_impact_report(
        baseline=dev_baseline,
        after=dev_after,
        tasks=cfg.dev_probe_tasks,
        output_path=output_dir / "dev_impact_report.json",
    )
    blind_report_rows = build_impact_report(
        baseline=blind_baseline,
        after=blind_after,
        tasks=cfg.blind_test_tasks,
        output_path=output_dir / "blind_impact_report.json",
    )
    (output_dir / "impact_report.json").write_text(json.dumps(blind_report_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    harness_files_after = list_harness_files(repo_root)
    metrics = build_benchmark_metrics(train_summary, dev_report_rows, harness_files_after, blind_impact_rows=blind_report_rows)
    (output_dir / "train_summary.json").write_text(json.dumps(train_summary, indent=2, ensure_ascii=False))
    (output_dir / "harness_files_after.json").write_text(json.dumps(harness_files_after, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    dev_improved = sum(1 for row in dev_report_rows if row["improved"])
    dev_regressed = sum(1 for row in dev_report_rows if row["regressed"])
    blind_improved = sum(1 for row in blind_report_rows if row["improved"])
    blind_regressed = sum(1 for row in blind_report_rows if row["regressed"])
    console.print(f"[bold]Dev impact:[/bold] improved={dev_improved}, regressed={dev_regressed}")
    console.print(f"[bold]Blind impact:[/bold] improved={blind_improved}, regressed={blind_regressed}")
    console.print(f"[bold]Blind report:[/bold] {output_dir / 'blind_impact_report.json'}")


async def _run_phase(
    cfg: BenchmarkConfig,
    phase: str,
    tasks: list[TaskConfig],
    weak: ChatClient,
    teacher: ChatClient,
    critic: ChatClient | None,
    teacher_system: str,
    critic_system: str,
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
        if request_teacher_diagnosis and benchmark_context and benchmark_context.get("patch_feedback"):
            result["context_patch_feedback"] = benchmark_context["patch_feedback"]
        diagnosis = parse_diagnosis(str(result["teacher_diagnosis_raw"])) if request_teacher_diagnosis else None
        if diagnosis is not None:
            result["teacher_diagnosis"] = diagnosis.model_dump()
        applied_patch_paths: list[str] = []
        if apply_patches and diagnosis is not None and diagnosis.patch_bundles and diagnosis.failure_categories:
            critic_policy_cases: dict[str, list[dict[str, object]]] = {}
            critic_audits: dict[str, object] = {}
            if _should_request_critic_cases(cfg.critic_mode, critic):
                for policy_name in _policy_names_from_patch_bundles(diagnosis.patch_bundles):
                    existing_policy_tests = _policy_tests_from_patch_bundles(diagnosis.patch_bundles, policy_name)
                    audit = await request_policy_audit_cases(
                        critic,
                        critic_system,
                        task,
                        diagnosis.patch_bundles,
                        policy_name,
                        existing_policy_tests,
                    )
                    critic_audits[policy_name] = audit
                    critic_policy_cases[policy_name] = list(audit.get("audit_cases", []))
            if critic_audits:
                result["critic_audits"] = critic_audits
            patch_result = apply_patch_bundles_atomically(
                repo_root,
                diagnosis.patch_bundles,
                task,
                diagnosis.harness_manifest,
                critic_policy_cases=critic_policy_cases,
            )
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
                "context_patch_feedback": result.get("context_patch_feedback"),
                "rejection_reason": result.get("rejection_reason"),
            }
        )
        results[task.id] = result
    (phase_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def _policy_names_from_patch_bundles(patch_bundles) -> list[str]:
    names = []
    for bundle in patch_bundles:
        parts = Path(bundle.target_path).parts
        if len(parts) == 3 and parts[0] == "harness" and parts[1] == "runtime_policies" and parts[2].endswith(".py"):
            names.append(Path(parts[2]).stem)
    return sorted(set(names))


def _critic_enabled(mode: str) -> bool:
    return mode == "always"


def _should_request_critic_cases(mode: str, critic: ChatClient | None) -> bool:
    if mode == "off":
        return False
    if mode == "always":
        return critic is not None
    raise ValueError(f"Unsupported critic_mode: {mode}")


def _policy_tests_from_patch_bundles(patch_bundles, policy_name: str) -> dict[str, object] | None:
    target = f"harness/tests/{policy_name}.json"
    for bundle in patch_bundles:
        if bundle.target_path != target:
            continue
        try:
            data = json.loads(bundle.content)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None
    return None


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
