from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console

from agentdistill.config import BenchmarkConfig, TaskConfig, load_benchmark_config
from agentdistill.critic import request_policy_audit_cases
from agentdistill.diagnosis import Diagnosis, parse_diagnosis, write_patch_artifact
from agentdistill.feedback import build_patch_feedback, build_transfer_feedback, merge_benchmark_context
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
    transfer_context = _initial_transfer_context(cfg, dev_baseline)
    patch_feedback: dict[str, object] | None = None
    transfer_feedback: dict[str, object] | None = None

    train_summary: list[dict[str, object]] = []
    for iteration in range(1, cfg.evolve_iterations + 1):
        phase = f"evolve_train_iter_{iteration:02d}"
        phase_kind = _phase_kind(cfg, patch_feedback)
        benchmark_context = _benchmark_context_for_iteration(transfer_context, patch_feedback, transfer_feedback, phase_kind)
        tasks = _tasks_for_evolve_iteration(cfg, patch_feedback, transfer_feedback)
        train_results = await _run_phase(
            cfg,
            phase=phase,
            tasks=tasks,
            weak=weak,
            teacher=teacher,
            critic=critic,
            teacher_system=teacher_system,
            critic_system=critic_system,
            output_dir=output_dir,
            apply_patches=True,
            repo_root=repo_root,
            benchmark_context=benchmark_context,
        )
        patch_feedback = build_patch_feedback(train_results, iteration)
        accepted_harness = any(result.get("patch_status") == "accepted" for result in train_results.values())
        train_summary.extend(
            {
                "iteration": iteration,
                "phase_kind": phase_kind,
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
                "context_transfer_feedback": result.get("context_transfer_feedback"),
                "inner_repair_attempts": result.get("inner_repair_attempts", []),
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
        transfer_feedback = build_transfer_feedback(
            cfg.dev_probe_tasks,
            dev_baseline,
            probe_results,
            iteration,
            accepted_harness=accepted_harness,
            previous_feedback=transfer_feedback,
        )

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
        if request_teacher_diagnosis and benchmark_context and benchmark_context.get("repair_mode") == "focused":
            result = await _run_focused_repair_task(task, teacher, teacher_system, weak_system, benchmark_context)
        else:
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
        if request_teacher_diagnosis and benchmark_context and benchmark_context.get("transfer_feedback"):
            result["context_transfer_feedback"] = benchmark_context["transfer_feedback"]
        diagnosis = parse_diagnosis(str(result["teacher_diagnosis_raw"])) if request_teacher_diagnosis else None
        if diagnosis is not None:
            result["teacher_diagnosis"] = diagnosis.model_dump()
        applied_patch_paths: list[str] = []
        if apply_patches and diagnosis is not None and diagnosis.patch_bundles and diagnosis.failure_categories:
            patch_result = await _apply_diagnosis_with_optional_audit(
                cfg,
                critic,
                critic_system,
                task,
                diagnosis,
                repo_root,
            )
            if patch_result.get("critic_audits"):
                result["critic_audits"] = patch_result["critic_audits"]
            result.update({key: value for key, value in patch_result.items() if key != "critic_audits"})
            if result.get("patch_status") == "rejected" and cfg.inner_repair_attempts > 0:
                repair_attempts = await _run_inner_repair_attempts(
                    cfg=cfg,
                    task=task,
                    original_result=result,
                    iteration_context=benchmark_context or {},
                    teacher=teacher,
                    teacher_system=teacher_system,
                    weak_system=weak_system,
                    critic=critic,
                    critic_system=critic_system,
                    repo_root=repo_root,
                    max_attempts=cfg.inner_repair_attempts,
                    phase_dir=phase_dir,
                )
                if repair_attempts:
                    result["inner_repair_attempts"] = repair_attempts
                    final_attempt = repair_attempts[-1]
                    if final_attempt.get("patch_status") == "accepted":
                        result.update(
                            {
                                key: value
                                for key, value in final_attempt.items()
                                if key
                                in {
                                    "teacher_diagnosis_raw",
                                    "teacher_diagnosis",
                                    "patch_status",
                                    "applied_patch_paths",
                                    "rejected_patch_paths",
                                    "contract_validation",
                                    "harness_manifest",
                                    "rejection_reason",
                                    "critic_audits",
                                }
                            }
                        )
                        result["rejection_reason"] = final_attempt.get("rejection_reason")
            applied_patch_paths = list(patch_result.get("applied_patch_paths", []))
            if result.get("patch_status") == "accepted":
                applied_patch_paths = list(result.get("applied_patch_paths", []))
            result["applied_patch_path"] = applied_patch_paths[0] if applied_patch_paths else None
            rejected_paths = list(patch_result.get("rejected_patch_paths", []))
            if result.get("patch_status") == "accepted":
                rejected_paths = list(result.get("rejected_patch_paths", []))
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
                "context_transfer_feedback": result.get("context_transfer_feedback"),
                "inner_repair_attempts": result.get("inner_repair_attempts", []),
                "rejection_reason": result.get("rejection_reason"),
            }
        )
        results[task.id] = result
    (phase_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


async def _run_focused_repair_task(
    task: TaskConfig,
    teacher: ChatClient,
    teacher_system: str,
    weak_system: str,
    benchmark_context: dict[str, object],
) -> dict[str, object]:
    messages = [
        {"role": "system", "content": teacher_system},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task_id": task.id,
                    "task_instruction": task.instruction,
                    "expected_answer": task.expected_answer,
                    "rubric": task.rubric,
                    "weak_system_prompt": weak_system,
                    "weak_answer": "",
                    "initial_weak_answer": "",
                    "tool_call": None,
                    "tool_result": None,
                    "runtime_policy_results": [],
                    "benchmark_context": benchmark_context,
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]
    return {
        "task_id": task.id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weak_answer": "",
        "initial_weak_answer": "",
        "tool_call": None,
        "tool_result": None,
        "runtime_policy_results": [],
        "focused_repair": True,
        "teacher_diagnosis_raw": await teacher.complete(messages, temperature=0.1),
    }


async def _apply_diagnosis_with_optional_audit(
    cfg: BenchmarkConfig,
    critic: ChatClient | None,
    critic_system: str,
    task: TaskConfig,
    diagnosis: Diagnosis,
    repo_root: Path,
) -> dict[str, object]:
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
    patch_result = apply_patch_bundles_atomically(
        repo_root,
        diagnosis.patch_bundles,
        task,
        diagnosis.harness_manifest,
        critic_policy_cases=critic_policy_cases,
    )
    if critic_audits:
        patch_result["critic_audits"] = critic_audits
    return patch_result


def _infer_repair_scope(patch_feedback: dict[str, Any]) -> dict[str, Any]:
    allowed_paths: set[str] = set()
    failure_kinds: set[str] = set()
    rejected_paths: list[str] = []

    for bundle in _as_dicts(patch_feedback.get("rejected_bundles")):
        for path in _as_strings(bundle.get("rejected_patch_paths")):
            normalized = _normalize_harness_path(path)
            if normalized is not None:
                rejected_paths.append(normalized)
        for contract in _as_dicts(bundle.get("failed_contracts")):
            contract_allowed, contract_kinds = _scope_from_failed_contract(contract)
            allowed_paths.update(contract_allowed)
            failure_kinds.update(contract_kinds)

    if not allowed_paths:
        allowed_paths.update(rejected_paths)
        if rejected_paths:
            failure_kinds.add("fallback_rejected_paths")

    return {
        "allowed_repair_paths": sorted(allowed_paths),
        "failure_kinds": sorted(failure_kinds),
        "source_rejected_paths": sorted(set(rejected_paths)),
        "scope_reason": _repair_scope_reason(failure_kinds),
    }


def _scope_from_failed_contract(contract: dict[str, Any]) -> tuple[set[str], set[str]]:
    allowed_paths: set[str] = set()
    failure_kinds: set[str] = set()

    path = _normalize_harness_path(contract.get("path"))
    if path is not None:
        kind, stem = _artifact_kind_and_stem(path)
        if kind == "tool":
            _add_tool_scope(allowed_paths, stem)
            failure_kinds.add("tool")
        elif kind == "runtime_policy":
            _add_policy_scope(allowed_paths, stem)
            failure_kinds.add("runtime_policy")
        elif kind == "test":
            allowed_paths.add(path)

    tool_name = contract.get("tool")
    if isinstance(tool_name, str) and tool_name:
        _add_tool_scope(allowed_paths, tool_name)
        failure_kinds.add("tool")

    policy_name = contract.get("policy")
    if isinstance(policy_name, str) and policy_name:
        _add_policy_scope(allowed_paths, policy_name)
        failure_kinds.add("runtime_policy")

    linked_tools = sorted(set(_extract_nested_tool_names(contract)))
    is_forced_tool_failure = bool(contract.get("policy_result")) and (
        bool(contract.get("tool_result")) or "forced tool" in str(contract.get("reason", "")).lower()
    )
    if is_forced_tool_failure and linked_tools:
        failure_kinds.add("tool_policy_pair")
        for linked_tool in linked_tools:
            _add_tool_scope(allowed_paths, linked_tool)

    return allowed_paths, failure_kinds


def _reject_out_of_scope_repair(
    diagnosis: Diagnosis,
    repair_scope: dict[str, Any] | None,
    repo_root: Path,
) -> dict[str, Any] | None:
    if not repair_scope:
        return None
    allowed_paths = set(_as_strings(repair_scope.get("allowed_repair_paths")))
    if not allowed_paths:
        return None

    target_paths = []
    out_of_scope = []
    for bundle in diagnosis.patch_bundles:
        normalized = _normalize_harness_path(bundle.target_path)
        target = normalized or bundle.target_path
        target_paths.append(target)
        if normalized not in allowed_paths:
            out_of_scope.append(target)

    if not out_of_scope:
        return None

    return {
        "patch_status": "rejected",
        "applied_patch_paths": [],
        "rejected_patch_paths": [_absolute_harness_path(repo_root, path) for path in target_paths],
        "contract_validation": [
            {
                "ok": False,
                "reason": "inner repair patch targets outside allowed repair scope",
                "allowed_repair_paths": sorted(allowed_paths),
                "out_of_scope_paths": out_of_scope,
            }
        ],
        "rejection_reason": "inner repair patch targets outside allowed repair scope",
        "harness_manifest": diagnosis.harness_manifest.model_dump() if diagnosis.harness_manifest is not None else None,
    }


def _add_tool_scope(paths: set[str], tool_name: str) -> None:
    paths.add(f"harness/tools/{tool_name}.py")
    paths.add(f"harness/tests/{tool_name}.json")


def _add_policy_scope(paths: set[str], policy_name: str) -> None:
    paths.add(f"harness/runtime_policies/{policy_name}.py")
    paths.add(f"harness/tests/{policy_name}.json")


def _artifact_kind_and_stem(path: str) -> tuple[str | None, str | None]:
    parts = Path(path).parts
    if len(parts) != 3 or parts[0] != "harness":
        return None, None
    stem = Path(parts[2]).stem
    if parts[1] == "tools" and parts[2].endswith(".py"):
        return "tool", stem
    if parts[1] == "runtime_policies" and parts[2].endswith(".py"):
        return "runtime_policy", stem
    if parts[1] == "tests" and parts[2].endswith(".json"):
        return "test", stem
    return None, None


def _normalize_harness_path(path: Any) -> str | None:
    if not isinstance(path, str) or not path:
        return None
    parts = Path(path).parts
    if "harness" not in parts:
        return None
    harness_index = parts.index("harness")
    normalized = "/".join(parts[harness_index:])
    if normalized.startswith("harness/"):
        return normalized
    return None


def _absolute_harness_path(repo_root: Path, path: str) -> str:
    normalized = _normalize_harness_path(path)
    if normalized is None:
        return path
    return str((repo_root / normalized).resolve())


def _extract_nested_tool_names(value: Any) -> list[str]:
    names: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "tool_name" and isinstance(item, str) and item:
                names.append(item)
            else:
                names.extend(_extract_nested_tool_names(item))
    elif isinstance(value, list):
        for item in value:
            names.extend(_extract_nested_tool_names(item))
    return names


def _as_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _as_strings(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _repair_scope_reason(failure_kinds: set[str]) -> str:
    if "tool_policy_pair" in failure_kinds:
        return "forced tool failures may require repairing only the linked runtime policy, tool, and their tests"
    if failure_kinds == {"tool"}:
        return "tool contract failures should repair only the tool and matching tests"
    if failure_kinds == {"runtime_policy"}:
        return "runtime policy contract failures should repair only the policy and matching tests"
    if "fallback_rejected_paths" in failure_kinds:
        return "no structured failing artifact was available, so repair is limited to rejected patch paths"
    return "repair is limited to artifacts directly named by failed contracts"


async def _run_inner_repair_attempts(
    cfg: BenchmarkConfig,
    task: TaskConfig,
    original_result: dict[str, object],
    iteration_context: dict[str, object],
    teacher: ChatClient,
    teacher_system: str,
    weak_system: str,
    critic: ChatClient | None,
    critic_system: str,
    repo_root: Path,
    max_attempts: int,
    phase_dir: Path,
) -> list[dict[str, object]]:
    attempts = []
    current_result = original_result
    for attempt_index in range(1, max_attempts + 1):
        patch_feedback = build_patch_feedback({task.id: current_result}, iteration=attempt_index)
        if not patch_feedback.get("has_rejections"):
            break
        repair_context = dict(iteration_context)
        repair_context["patch_feedback"] = patch_feedback
        repair_context["repair_scope"] = _infer_repair_scope(patch_feedback)
        repair_context["repair_mode"] = "focused"
        repair_task = _build_focused_repair_task(
            patch_feedback,
            repair_context.get("transfer_feedback"),
            repair_context.get("repair_scope") if isinstance(repair_context.get("repair_scope"), dict) else None,
        )
        repair_result = await _run_focused_repair_task(repair_task, teacher, teacher_system, weak_system, repair_context)
        repair_result["inner_repair_attempt"] = attempt_index
        repair_result["context_patch_feedback"] = patch_feedback
        repair_result["context_repair_scope"] = repair_context["repair_scope"]
        if repair_context.get("transfer_feedback"):
            repair_result["context_transfer_feedback"] = repair_context["transfer_feedback"]
        diagnosis = parse_diagnosis(str(repair_result["teacher_diagnosis_raw"]))
        repair_result["teacher_diagnosis"] = diagnosis.model_dump()
        if diagnosis.patch_bundles and diagnosis.failure_categories:
            patch_result = _reject_out_of_scope_repair(diagnosis, repair_context["repair_scope"], repo_root)
            if patch_result is None:
                patch_result = await _apply_diagnosis_with_optional_audit(
                    cfg,
                    critic,
                    critic_system,
                    repair_task,
                    diagnosis,
                    repo_root,
                )
            repair_result.update(patch_result)
        else:
            repair_result["patch_status"] = "skipped"
        write_patch_artifact(phase_dir / "patches", f"{task.id}-inner-repair-{attempt_index}", cfg.name, diagnosis)
        (phase_dir / f"{task.id}.inner_repair_{attempt_index}.json").write_text(
            json.dumps(repair_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        attempts.append(repair_result)
        current_result = repair_result
        if repair_result.get("patch_status") == "accepted":
            break
    return attempts


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
                "task_instruction": task.instruction,
                "expected_answer": task.expected_answer,
                "rubric": task.rubric,
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


def _initial_transfer_context(cfg: BenchmarkConfig, dev_baseline: dict[str, dict[str, object]]) -> dict[str, object]:
    if cfg.transfer_context_mode == "heldout_probe":
        return _build_transfer_context(cfg.dev_probe_tasks, dev_baseline)
    if cfg.transfer_context_mode == "feedback_only":
        return {"heldout_probe": []}
    raise ValueError(f"Unsupported transfer_context_mode: {cfg.transfer_context_mode}")


def _phase_kind(cfg: BenchmarkConfig, patch_feedback: dict[str, object] | None) -> str:
    if cfg.repair_mode == "focused" and patch_feedback and patch_feedback.get("has_rejections"):
        return "focused_repair"
    return "full_train"


def _tasks_for_evolve_iteration(
    cfg: BenchmarkConfig,
    patch_feedback: dict[str, object] | None,
    transfer_feedback: dict[str, object] | None,
) -> list[TaskConfig]:
    if _phase_kind(cfg, patch_feedback) != "focused_repair":
        return cfg.train_tasks
    return [_build_focused_repair_task(patch_feedback or {}, transfer_feedback)]


def _build_focused_repair_task(
    patch_feedback: dict[str, object],
    transfer_feedback: dict[str, object] | None,
    repair_scope: dict[str, object] | None = None,
) -> TaskConfig:
    rejected = patch_feedback.get("rejected_bundles", [])
    first_rejected = rejected[0] if isinstance(rejected, list) and rejected else {}
    failed_contracts = first_rejected.get("failed_contracts", []) if isinstance(first_rejected, dict) else []
    failed_paths = first_rejected.get("rejected_patch_paths", []) if isinstance(first_rejected, dict) else []
    bundle_id = first_rejected.get("bundle_id") if isinstance(first_rejected, dict) else None
    transfer_tasks = (transfer_feedback or {}).get("failed_tasks", []) if isinstance(transfer_feedback, dict) else []
    first_transfer = transfer_tasks[0] if isinstance(transfer_tasks, list) and transfer_tasks else {}
    expected_answer = first_transfer.get("expected_answer") if isinstance(first_transfer, dict) else None
    task_instruction = first_transfer.get("task_instruction") if isinstance(first_transfer, dict) else None
    instruction = {
        "repair_mode": "focused",
        "objective": "Repair the rejected harness bundle/artifacts. Do not solve a new user task. Preserve the original transfer repair intent while fixing the failed contract.",
        "rejected_bundle_id": bundle_id,
        "rejected_patch_paths": failed_paths,
        "failed_contracts": failed_contracts,
        "repair_scope": repair_scope or {},
        "representative_transfer_failure": first_transfer,
    }
    return TaskConfig(
        id="focused_repair",
        instruction=json.dumps(instruction, ensure_ascii=False, indent=2),
        expected_answer=expected_answer if isinstance(expected_answer, str) else None,
        rubric=(
            "Repair only the rejected harness artifacts needed to satisfy failed contracts and unresolved transfer failures. "
            "Prefer preserving bundle intent over inventing a new architecture."
            + (f"\nRepresentative failed transfer task:\n{task_instruction}" if isinstance(task_instruction, str) else "")
        ),
    )


def _benchmark_context_for_iteration(
    transfer_context: dict[str, object],
    patch_feedback: dict[str, object] | None,
    transfer_feedback: dict[str, object] | None,
    phase_kind: str,
) -> dict[str, object]:
    context = merge_benchmark_context(transfer_context, patch_feedback, transfer_feedback)
    if phase_kind == "focused_repair":
        context["repair_mode"] = "focused"
    return context


if __name__ == "__main__":
    app()
