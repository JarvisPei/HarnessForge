from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console

from agentdistill.benchmark import _build_focused_repair_task, _infer_repair_scope, _reject_out_of_scope_repair, _run_focused_repair_task
from agentdistill.config import TaskConfig, load_benchmark_config
from agentdistill.diagnosis import PatchBundle, parse_diagnosis
from agentdistill.feedback import build_patch_feedback
from agentdistill.manifest import HarnessManifest
from agentdistill.models import ChatClient, load_model_settings
from agentdistill.patches import apply_patch_bundles_atomically
from agentdistill.repair_fixture import build_repair_fixture_case
from agentdistill.report import build_impact_report
from agentdistill.run import run_task
from agentdistill.tools import RuntimePolicyRegistry, ToolRegistry


app = typer.Typer(add_completion=False)
console = Console()


@dataclass(frozen=True)
class RepairFamilyCase:
    case_id: str
    task: TaskConfig
    bad_policy_bundles: list[PatchBundle]
    good_policy_bundles: list[PatchBundle]
    manifest: HarnessManifest | None


@app.command()
def main(
    output_dir: Path = typer.Option(Path("outputs/repair_family"), "--output-dir", "-o"),
    profile: str | None = typer.Option(None, "--profile", "-p"),
) -> None:
    load_dotenv(override=True)
    try:
        report = asyncio.run(run_repair_family(output_dir, profile))
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(json.dumps(report, indent=2, ensure_ascii=False))

async def run_repair_family(
    output_dir: Path,
    profile: str | None,
    *,
    repo_root: Path | None = None,
    teacher: ChatClient | None = None,
    weak: ChatClient | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = repo_root or Path(__file__).resolve().parent.parent
    transfer_cfg = load_benchmark_config(repo_root / "configs/benchmark_repair_mechanism.yaml")
    teacher = teacher or ChatClient(load_model_settings("teacher", profile))
    weak = weak or ChatClient(load_model_settings("weak", profile))
    teacher_system = (repo_root / "prompts/teacher_diagnosis.md").read_text().strip()
    cases = build_repair_family_cases()

    case_reports = []
    for case in cases:
        case_dir = output_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        report = await _run_case(
            repo_root=repo_root,
            case=case,
            case_dir=case_dir,
            transfer_cfg=transfer_cfg,
            weak=weak,
            teacher=teacher,
            teacher_system=teacher_system,
        )
        case_reports.append(report)

    summary = {
        "cases": len(case_reports),
        "repair_successes": sum(1 for report in case_reports if report["repair_success"]),
        "dev_improved": sum(report["dev_transfer"]["improved"] for report in case_reports),
        "blind_improved": sum(report["blind_transfer"]["improved"] for report in case_reports),
    }
    report = {"cases": case_reports, "summary": summary}
    (output_dir / "repair_family_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_repair_family_cases() -> list[RepairFamilyCase]:
    fixture = build_repair_fixture_case()
    fallback_task = TaskConfig(
        id="fallback_rejected_paths",
        instruction="Repair the rejected harness guideline by making the note precise and reusable.",
        expected_answer=None,
    )
    fallback_bad = [
        PatchBundle(
            target_path="harness/guidelines/fallback_repair_note.md",
            action="append",
            content="# Fallback repair note\nUse the rejected path itself as the repair scope.",
        )
    ]
    fallback_good = [
        PatchBundle(
            target_path="harness/guidelines/fallback_repair_note.md",
            action="create_or_replace",
            content="# Fallback repair note\nUse the rejected path itself as the repair scope.\n",
        )
    ]
    return [
        RepairFamilyCase(
            case_id="tool_policy_pair",
            task=fixture.task,
            bad_policy_bundles=fixture.bad_policy_bundles,
            good_policy_bundles=fixture.good_policy_bundles,
            manifest=fixture.manifest,
        ),
        RepairFamilyCase(
            case_id="fallback_rejected_paths",
            task=fallback_task,
            bad_policy_bundles=fallback_bad,
            good_policy_bundles=fallback_good,
            manifest=None,
        ),
    ]


async def _run_case(
    repo_root: Path,
    case: RepairFamilyCase,
    case_dir: Path,
    transfer_cfg,
    weak: ChatClient,
    teacher: ChatClient,
    teacher_system: str,
) -> dict[str, Any]:
    workspace_root = Path(tempfile.mkdtemp(prefix=f"repair_family_{case.case_id}_"))
    try:
        _copy_workspace(repo_root, workspace_root)
        weak_system = _weak_system(workspace_root)
        baseline = await _run_transfer_suite(workspace_root, transfer_cfg.dev_probe_tasks + transfer_cfg.blind_test_tasks, weak, teacher, teacher_system, weak_system)
        seed_result = apply_patch_bundles_atomically(workspace_root, case.bad_policy_bundles, case.task, case.manifest)
        patch_feedback = build_patch_feedback({case.task.id: seed_result}, iteration=1)
        repair_scope = _infer_repair_scope(patch_feedback)
        repair_task = _build_focused_repair_task(patch_feedback, None, repair_scope)
        repair_context = {"repair_mode": "focused", "patch_feedback": patch_feedback, "repair_scope": repair_scope}
        repair_run = await _run_focused_repair_task(repair_task, teacher, teacher_system, weak_system, repair_context)
        diagnosis = parse_diagnosis(str(repair_run["teacher_diagnosis_raw"]))
        repair_run["teacher_diagnosis"] = diagnosis.model_dump()
        out_of_scope = _reject_out_of_scope_repair(diagnosis, repair_scope, workspace_root)
        if out_of_scope is not None:
            final_patch = out_of_scope
        elif diagnosis.patch_bundles and diagnosis.failure_categories:
            final_patch = apply_patch_bundles_atomically(workspace_root, diagnosis.patch_bundles, case.task, diagnosis.harness_manifest)
        else:
            final_patch = {
                "patch_status": "skipped",
                "applied_patch_paths": [],
                "rejected_patch_paths": [],
                "rejection_reason": "teacher did not produce a repair patch",
                "contract_validation": [],
                "harness_manifest": diagnosis.harness_manifest.model_dump() if diagnosis.harness_manifest is not None else None,
            }
        after = await _run_transfer_suite(workspace_root, transfer_cfg.dev_probe_tasks + transfer_cfg.blind_test_tasks, weak, teacher, teacher_system, _weak_system(workspace_root))
        dev_report = build_impact_report(
            {task.id: baseline[task.id] for task in transfer_cfg.dev_probe_tasks},
            {task.id: after[task.id] for task in transfer_cfg.dev_probe_tasks},
            transfer_cfg.dev_probe_tasks,
            case_dir / "dev_impact_report.json",
        )
        blind_report = build_impact_report(
            {task.id: baseline[task.id] for task in transfer_cfg.blind_test_tasks},
            {task.id: after[task.id] for task in transfer_cfg.blind_test_tasks},
            transfer_cfg.blind_test_tasks,
            case_dir / "blind_impact_report.json",
        )
        case_report = {
            "case_id": case.case_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed_patch_result": seed_result,
            "patch_feedback": patch_feedback,
            "repair_scope": repair_scope,
            "teacher_diagnosis": repair_run["teacher_diagnosis"],
            "final_patch_result": final_patch,
            "repair_run": repair_run,
            "repair_success": final_patch.get("patch_status") == "accepted",
            "repair_success_via": "scoped_inner_repair" if final_patch.get("patch_status") == "accepted" else "none",
            "scoped_inner_repair_success": final_patch.get("patch_status") == "accepted",
            "dev_transfer": _transfer_summary(dev_report),
            "blind_transfer": _transfer_summary(blind_report),
        }
        (case_dir / "case_report.json").write_text(json.dumps(case_report, indent=2, ensure_ascii=False), encoding="utf-8")
        return case_report
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)


async def _run_transfer_suite(
    workspace_root: Path,
    tasks: list[TaskConfig],
    weak: ChatClient,
    teacher: ChatClient,
    teacher_system: str,
    weak_system: str,
) -> dict[str, dict[str, Any]]:
    tools = ToolRegistry(workspace_root / "harness" / "tools")
    policies = RuntimePolicyRegistry(workspace_root / "harness" / "runtime_policies")
    results: dict[str, dict[str, Any]] = {}
    for task in tasks:
        result = await run_task(
            task,
            weak,
            teacher,
            weak_system,
            teacher_system,
            tools,
            policies,
            request_teacher_diagnosis=False,
        )
        results[task.id] = result
    return results


def _transfer_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "tasks": len(rows),
        "before_success": sum(1 for row in rows if row.get("before_success") is True),
        "after_success": sum(1 for row in rows if row.get("after_success") is True),
        "improved": sum(1 for row in rows if row.get("improved") is True),
        "regressed": sum(1 for row in rows if row.get("regressed") is True),
    }


def _copy_workspace(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", ".venv", "outputs", "__pycache__", "*.pyc", ".pytest_cache"))


def _weak_system(workspace_root: Path) -> str:
    return (workspace_root / "prompts/weak_system.md").read_text().strip()


if __name__ == "__main__":
    app()
