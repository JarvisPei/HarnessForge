from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console

from agentdistill.benchmark import _build_focused_repair_task, _infer_repair_scope, _reject_out_of_scope_repair, _run_focused_repair_task
from agentdistill.config import TaskConfig
from agentdistill.diagnosis import parse_diagnosis
from agentdistill.feedback import build_patch_feedback, merge_benchmark_context
from agentdistill.models import ChatClient, load_model_settings
from agentdistill.patches import apply_patch_bundles_atomically, patch_group_is_executable
from agentdistill.prompt_loader import load_teacher_system_prompt
from agentdistill.repair_fixture import build_repair_fixture_case


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    output_dir: Path = typer.Option(Path("outputs/repair_probe"), "--output-dir", "-o"),
    profile: str | None = typer.Option(None, "--profile", "-p"),
) -> None:
    load_dotenv(override=True)
    try:
        report = asyncio.run(run_repair_probe(output_dir, profile))
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(json.dumps(report, indent=2, ensure_ascii=False))


async def run_repair_probe(
    output_dir: Path,
    profile: str | None,
    *,
    teacher: Any | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = repo_root or Path(__file__).resolve().parent.parent
    case = build_repair_fixture_case()
    teacher = teacher or ChatClient(load_model_settings("teacher", profile))
    teacher_system = load_teacher_system_prompt(repo_root)
    weak_system = (repo_root / "prompts/weak_system.md").read_text().strip()

    seed_result = apply_patch_bundles_atomically(repo_root, case.bad_policy_bundles, case.task, case.manifest)
    patch_feedback = build_patch_feedback({case.task.id: seed_result}, iteration=1)
    repair_scope = _infer_repair_scope(patch_feedback)
    repair_task = _build_focused_repair_task(patch_feedback, None, repair_scope)
    benchmark_context = merge_benchmark_context({"heldout_probe": []}, patch_feedback)
    repair_context = {
        "repair_mode": "focused",
        "patch_feedback": patch_feedback,
        "repair_scope": repair_scope,
    }
    repair_run = await _run_focused_repair_task(
        repair_task,
        teacher,
        teacher_system,
        weak_system,
        {
            **benchmark_context,
            **repair_context,
        },
    )
    diagnosis = parse_diagnosis(str(repair_run["teacher_diagnosis_raw"]))
    repair_run["teacher_diagnosis"] = diagnosis.model_dump()

    out_of_scope = _reject_out_of_scope_repair(diagnosis, repair_scope, repo_root)
    if out_of_scope is not None:
        final_patch = out_of_scope
    elif diagnosis.patch_bundles and diagnosis.failure_categories:
        final_patch = apply_patch_bundles_atomically(
            repo_root,
            diagnosis.patch_bundles,
            case.task,
            diagnosis.harness_manifest,
            teacher_policy_cases=diagnosis.policy_audit_cases if patch_group_is_executable(diagnosis.patch_bundles) else None,
            teacher_tool_cases=diagnosis.tool_audit_cases if patch_group_is_executable(diagnosis.patch_bundles) else None,
        )
    else:
        final_patch = {
            "patch_status": "skipped",
            "applied_patch_paths": [],
            "rejected_patch_paths": [],
            "rejection_reason": "teacher did not produce a repair patch",
            "contract_validation": [],
            "harness_manifest": diagnosis.harness_manifest.model_dump() if diagnosis.harness_manifest is not None else None,
        }

    report = {
        "task_id": case.task.id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed_patch_result": seed_result,
        "patch_feedback": patch_feedback,
        "repair_scope": repair_scope,
        "repair_context": repair_context,
        "teacher_diagnosis": repair_run["teacher_diagnosis"],
        "repair_run": repair_run,
        "final_patch_result": final_patch,
        "repair_success": final_patch.get("patch_status") == "accepted",
        "repair_success_via": "scoped_inner_repair" if final_patch.get("patch_status") == "accepted" else "none",
        "scoped_inner_repair_success": final_patch.get("patch_status") == "accepted",
    }
    output_path = output_dir / "repair_probe.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    app()
