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
from agentdistill.feedback import build_patch_feedback, build_transfer_feedback
from agentdistill.harness import load_system_prompt
from agentdistill.manifest import HarnessManifest, ManifestArtifact
from agentdistill.models import ChatClient, load_model_settings
from agentdistill.patches import apply_patch_bundles_atomically, patch_group_is_executable
from agentdistill.repair_fixture import build_repair_fixture_case
from agentdistill.report import build_impact_report, evaluate_success
from agentdistill.run import run_task
from agentdistill.tools import RuntimePolicyRegistry, ToolRegistry


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


@dataclass(frozen=True)
class RepairFamilyCase:
    case_id: str
    mechanism: str
    task: TaskConfig
    bad_policy_bundles: list[PatchBundle]
    good_policy_bundles: list[PatchBundle]
    manifest: HarnessManifest | None
    dev_probe_tasks: list[TaskConfig] | None = None
    blind_test_tasks: list[TaskConfig] | None = None
    mechanism_only: bool = False
    repair_scope_override: dict[str, Any] | None = None
    diagnostic: bool = False


@app.command()
def main(
    output_dir: Path = typer.Option(Path("outputs/repair_family"), "--output-dir", "-o"),
    profile: str | None = typer.Option(None, "--profile", "-p"),
    include_diagnostics: bool = typer.Option(False, "--include-diagnostics"),
    filter_baseline_probes: bool = typer.Option(False, "--filter-baseline-probes"),
    transfer_tight: bool = typer.Option(False, "--transfer-tight"),
    transfer_feedback_repair: bool = typer.Option(False, "--transfer-feedback-repair"),
) -> None:
    load_dotenv(override=True)
    try:
        if filter_baseline_probes:
            report = asyncio.run(
                run_probe_filter(output_dir, profile, include_diagnostics=include_diagnostics)
            )
        else:
            report = asyncio.run(
                run_repair_family(
                    output_dir,
                    profile,
                    include_diagnostics=include_diagnostics,
                    transfer_tight=transfer_tight,
                    transfer_feedback_repair=transfer_feedback_repair,
                )
            )
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(json.dumps(report, indent=2, ensure_ascii=False))


async def run_probe_filter(
    output_dir: Path,
    profile: str | None,
    *,
    repo_root: Path | None = None,
    teacher: ChatClient | None = None,
    weak: ChatClient | None = None,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = repo_root or Path(__file__).resolve().parent.parent
    teacher = teacher or ChatClient(load_model_settings("teacher", profile))
    weak = weak or ChatClient(load_model_settings("weak", profile))
    teacher_system = (repo_root / "prompts/teacher_diagnosis.md").read_text().strip()
    cases = build_probe_filter_cases(include_diagnostics=include_diagnostics)
    workspace_parent = Path(tempfile.mkdtemp(prefix="repair_family_probe_filter_"))
    workspace_root = workspace_parent / "workspace"
    try:
        _copy_workspace(repo_root, workspace_root)
        weak_system = _weak_system(workspace_root)
        case_reports = []
        for case in cases:
            console.print(f"[bold]Filtering baseline probes:[/bold] {case.case_id}")
            report = await _run_probe_filter_case(
                case=case,
                case_dir=output_dir / case.case_id,
                workspace_root=workspace_root,
                weak=weak,
                teacher=teacher,
                teacher_system=teacher_system,
                weak_system=weak_system,
            )
            case_reports.append(report)
    finally:
        shutil.rmtree(workspace_parent, ignore_errors=True)

    summary = {
        "cases": len(case_reports),
        "mechanism_only": sum(1 for report in case_reports if report.get("mechanism_only") is True),
        "candidate_tasks": sum(report.get("baseline", {}).get("tasks", 0) for report in case_reports),
        "baseline_pass": sum(report.get("baseline", {}).get("passed", 0) for report in case_reports),
        "baseline_fail": sum(report.get("baseline", {}).get("failed", 0) for report in case_reports),
        "recommended_dev_candidates": sum(
            len(report.get("recommended_dev_task_ids", [])) for report in case_reports
        ),
        "recommended_blind_candidates": sum(
            len(report.get("recommended_blind_task_ids", [])) for report in case_reports
        ),
    }
    report = {"cases": case_reports, "summary": summary}
    (output_dir / "probe_filter_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report

async def run_repair_family(
    output_dir: Path,
    profile: str | None,
    *,
    repo_root: Path | None = None,
    teacher: ChatClient | None = None,
    weak: ChatClient | None = None,
    include_diagnostics: bool = False,
    transfer_tight: bool = False,
    transfer_feedback_repair: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = repo_root or Path(__file__).resolve().parent.parent
    transfer_cfg = load_benchmark_config(repo_root / "configs/benchmark_repair_mechanism.yaml")
    teacher = teacher or ChatClient(load_model_settings("teacher", profile))
    weak = weak or ChatClient(load_model_settings("weak", profile))
    teacher_system = (repo_root / "prompts/teacher_diagnosis.md").read_text().strip()
    cases = (
        build_probe_filter_cases(include_diagnostics=include_diagnostics)
        if transfer_tight
        else build_repair_family_cases(include_diagnostics=include_diagnostics)
    )

    case_reports = []
    for case in cases:
        case_dir = output_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold]Running repair family case:[/bold] {case.case_id}")
        try:
            report = await _run_case(
                repo_root=repo_root,
                case=case,
                case_dir=case_dir,
                transfer_cfg=transfer_cfg,
                weak=weak,
                teacher=teacher,
                teacher_system=teacher_system,
                transfer_feedback_repair=transfer_feedback_repair,
            )
        except Exception as exc:
            report = {
                "case_id": case.case_id,
                "mechanism": case.mechanism,
                "diagnostic": case.diagnostic,
                "mechanism_only": case.mechanism_only,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "repair_success": False,
                "repair_success_via": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            (case_dir / "case_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            console.print(f"[bold red]Case failed:[/bold red] {case.case_id}: {type(exc).__name__}: {exc}")
        case_reports.append(report)

    summary = {
        "cases": len(case_reports),
        "transfer_tight": transfer_tight,
        "transfer_feedback_repair": transfer_feedback_repair,
        "mechanisms": sorted({str(report.get("mechanism")) for report in case_reports if report.get("mechanism")}),
        "diagnostic_cases": sum(1 for report in case_reports if report.get("diagnostic") is True),
        "repair_successes": sum(1 for report in case_reports if report["repair_success"]),
        "errors": sum(1 for report in case_reports if "error_type" in report),
        "mechanism_only": sum(1 for report in case_reports if report.get("mechanism_only") is True),
        "dev_improved": sum(report.get("dev_transfer", {}).get("improved", 0) for report in case_reports),
        "blind_improved": sum(report.get("blind_transfer", {}).get("improved", 0) for report in case_reports),
        "dev_regressed": sum(report.get("dev_transfer", {}).get("regressed", 0) for report in case_reports),
        "blind_regressed": sum(report.get("blind_transfer", {}).get("regressed", 0) for report in case_reports),
    }
    report = {"cases": case_reports, "summary": summary}
    (output_dir / "repair_family_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_repair_family_cases(include_diagnostics: bool = False) -> list[RepairFamilyCase]:
    fixture = build_repair_fixture_case()
    fixture_dev_probes = [
        TaskConfig(
            id="dev_fixture_signed_updates",
            instruction="Use the signed updates: start=200, updates=[+15, -9]. Return the final count.",
            expected_answer="206",
            rubric="Correct computation is 200 + 15 - 9 = 206.",
        )
    ]
    fixture_blind_probes = [
        TaskConfig(
            id="blind_fixture_signed_updates",
            instruction="Use the signed updates: start=350, updates=[-40, +22, -8]. Return the final count.",
            expected_answer="324",
            rubric="Correct computation is 350 - 40 + 22 - 8 = 324.",
        )
    ]
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
    tool_task = TaskConfig(
        id="tool_signed_sum",
        instruction="Repair a signed-sum tool so it handles negative updates correctly.",
        expected_answer="118",
    )
    tool_dev_probes = [
        TaskConfig(
            id="dev_signed_sum_tool_updates",
            instruction=(
                "Use the signed_sum tool if it is available. Compute the final count for "
                'start=240 and updates=["+31", "-18", "-42", "+9", "-7"]. '
                "Return the final number with one short explanation sentence."
            ),
            expected_answer="213",
            rubric="Correct computation is 240 + 31 - 18 - 42 + 9 - 7 = 213.",
        )
    ]
    tool_blind_probes = [
        TaskConfig(
            id="blind_signed_sum_tool_updates",
            instruction=(
                "Use the signed_sum tool if it is available. Compute the final count for "
                'start=1000 and updates=["+125", "-240", "+16", "-33"]. '
                "Return the final number with one short explanation sentence."
            ),
            expected_answer="868",
            rubric="Correct computation is 1000 + 125 - 240 + 16 - 33 = 868.",
        )
    ]
    tool_bad = [
        PatchBundle(
            target_path="harness/tools/signed_sum.py",
            action="create_or_replace",
            content="""
def run(input: dict) -> dict:
    total = int(input.get("start", 0))
    for value in input.get("updates", []):
        text = str(value).strip().replace(",", "")
        total += abs(int(text))
    return {"ok": True, "result": total}
""".strip(),
        ),
        PatchBundle(
            target_path="harness/tests/signed_sum.json",
            action="create_or_replace",
            content="""
{
  "tool": "signed_sum",
  "cases": [
    {
      "input": {"start": 100, "updates": ["+25", "-7"]},
      "expected": {"ok": true, "result": 118}
    }
  ]
}
""".strip(),
        ),
    ]
    tool_good = [
        PatchBundle(
            target_path="harness/tools/signed_sum.py",
            action="create_or_replace",
            content="""
def run(input: dict) -> dict:
    total = int(input.get("start", 0))
    for value in input.get("updates", []):
        text = str(value).strip().replace(",", "")
        total += int(text)
    return {"ok": True, "result": total}
""".strip(),
        ),
        PatchBundle(
            target_path="harness/tests/signed_sum.json",
            action="create_or_replace",
            content="""
{
  "tool": "signed_sum",
  "cases": [
    {
      "input": {"start": 100, "updates": ["+25", "-7"]},
      "expected": {"ok": true, "result": 118}
    },
    {
      "input": {"start": 10, "updates": ["-3", "+8"]},
      "expected": {"ok": true, "result": 15}
    }
  ]
}
""".strip(),
        ),
    ]
    cases = [
        RepairFamilyCase(
            case_id="tool_policy_pair",
            mechanism="tool_policy_pair",
            task=fixture.task,
            bad_policy_bundles=fixture.bad_policy_bundles,
            good_policy_bundles=fixture.good_policy_bundles,
            manifest=fixture.manifest,
            dev_probe_tasks=fixture_dev_probes,
            blind_test_tasks=fixture_blind_probes,
        ),
        RepairFamilyCase(
            case_id="tool_contract_repair",
            mechanism="tool",
            task=tool_task,
            bad_policy_bundles=tool_bad,
            good_policy_bundles=tool_good,
            manifest=_manifest(["harness/tools/signed_sum.py", "harness/tests/signed_sum.json"], bundle_id="signed_sum_tool"),
            dev_probe_tasks=tool_dev_probes,
            blind_test_tasks=tool_blind_probes,
        ),
        RepairFamilyCase(
            case_id="fallback_rejected_paths",
            mechanism="prompt_guideline",
            task=fallback_task,
            bad_policy_bundles=fallback_bad,
            good_policy_bundles=fallback_good,
            manifest=None,
            mechanism_only=True,
        ),
    ]
    if include_diagnostics:
        cases.append(
            RepairFamilyCase(
                case_id="tool_policy_pair_scoped",
                mechanism="runtime_policy",
                task=fixture.task,
                bad_policy_bundles=fixture.bad_policy_bundles,
                good_policy_bundles=fixture.good_policy_bundles,
                manifest=fixture.manifest,
                dev_probe_tasks=fixture_dev_probes,
                blind_test_tasks=fixture_blind_probes,
                diagnostic=True,
                repair_scope_override={
                    "allowed_repair_paths": [
                        "harness/runtime_policies/force_fixture.py",
                        "harness/tests/force_fixture.json",
                    ],
                    "failure_kinds": ["runtime_policy"],
                    "source_rejected_paths": [
                        "harness/runtime_policies/force_fixture.py",
                        "harness/tests/force_fixture.json",
                    ],
                    "scope_reason": "diagnostic fixture constrains repair to the rejected runtime policy and its matching test",
                },
            )
        )
    return cases


def build_probe_filter_cases(include_diagnostics: bool = False) -> list[RepairFamilyCase]:
    cases = build_repair_family_cases(include_diagnostics=include_diagnostics)
    candidate_by_id = {
        "tool_policy_pair": (
            _candidate_signed_update_dev_tasks(),
            _candidate_signed_update_blind_tasks(),
        ),
        "tool_contract_repair": (
            _candidate_tool_dev_tasks(),
            _candidate_tool_blind_tasks(),
        ),
        "tool_policy_pair_scoped": (
            _candidate_signed_update_dev_tasks(),
            _candidate_signed_update_blind_tasks(),
        ),
    }
    filtered_cases = []
    for case in cases:
        dev_tasks, blind_tasks = candidate_by_id.get(case.case_id, (case.dev_probe_tasks, case.blind_test_tasks))
        filtered_cases.append(
            RepairFamilyCase(
                case_id=case.case_id,
                mechanism=case.mechanism,
                task=case.task,
                bad_policy_bundles=case.bad_policy_bundles,
                good_policy_bundles=case.good_policy_bundles,
                manifest=case.manifest,
                dev_probe_tasks=dev_tasks,
                blind_test_tasks=blind_tasks,
                mechanism_only=case.mechanism_only,
                repair_scope_override=case.repair_scope_override,
                diagnostic=case.diagnostic,
            )
        )
    return filtered_cases


def _candidate_signed_update_dev_tasks() -> list[TaskConfig]:
    return [
        TaskConfig(
            id="dev_filter_signed_dense_log",
            instruction=(
                "Compute the final count exactly. Start at 9,127 tokens. Apply signed updates in order: "
                "+384, -219, +77, -1,406, +502, -908, +116, -73, +641, -222, +39, -850. "
                "Return only the final integer."
            ),
            expected_answer="7208",
            rubric="Correct computation is 9127 + 384 - 219 + 77 - 1406 + 502 - 908 + 116 - 73 + 641 - 222 + 39 - 850 = 7208.",
        ),
        TaskConfig(
            id="dev_filter_signed_table_noise",
            instruction=(
                "A ledger starts with 15,000 units. Use only rows whose status is POSTED and ignore DRAFT rows. "
                "POSTED deltas: -1,275, +386, -942, +711, -208, +64, -530, +119, -76, +403, -999, +250. "
                "Return only the final integer."
            ),
            expected_answer="12903",
            rubric="Correct computation is 15000 - 1275 + 386 - 942 + 711 - 208 + 64 - 530 + 119 - 76 + 403 - 999 + 250 = 12903.",
        ),
        TaskConfig(
            id="dev_filter_signed_small_offsets",
            instruction=(
                "Start=731. Updates=[+44, -18, +27, -96, +105, -12, -58, +33, -21, +14]. "
                "Return the final value and no explanation."
            ),
            expected_answer="749",
            rubric="Correct computation is 731 + 44 - 18 + 27 - 96 + 105 - 12 - 58 + 33 - 21 + 14 = 749.",
        ),
    ]


def _candidate_signed_update_blind_tasks() -> list[TaskConfig]:
    return [
        TaskConfig(
            id="blind_filter_signed_reordered",
            instruction=(
                "Calculate the exact final balance. Initial balance: 42,000. "
                "Deltas: -811, +233, -1,440, +950, -72, -602, +315, -49, +128, -207, +88, -999. "
                "Return only the integer."
            ),
            expected_answer="39534",
            rubric="Correct computation is 42000 - 811 + 233 - 1440 + 950 - 72 - 602 + 315 - 49 + 128 - 207 + 88 - 999 = 39534.",
        ),
        TaskConfig(
            id="blind_filter_signed_mixed_units",
            instruction=(
                "Inventory starts at 6,550 packets. Apply adjustments: +120, -450, +315, -90, -700, +42, +58, -16, +900, -111. "
                "Return only the final packet count as an integer."
            ),
            expected_answer="6618",
            rubric="Correct computation is 6550 + 120 - 450 + 315 - 90 - 700 + 42 + 58 - 16 + 900 - 111 = 6618.",
        ),
        TaskConfig(
            id="blind_filter_signed_large_offsets",
            instruction=(
                "From 123,456 apply these signed changes: -9,876; +5,432; -2,100; +809; -77; -650; +1,200; -333; +444; -555. "
                "Return only the final integer."
            ),
            expected_answer="117750",
            rubric="Correct computation is 123456 - 9876 + 5432 - 2100 + 809 - 77 - 650 + 1200 - 333 + 444 - 555 = 117750.",
        ),
    ]


def _candidate_tool_dev_tasks() -> list[TaskConfig]:
    return [
        TaskConfig(
            id="dev_filter_tool_dense_updates",
            instruction=(
                "Use the signed_sum tool if it is available. Compute start=9,127 with "
                'updates=["+384", "-219", "+77", "-1,406", "+502", "-908", "+116", "-73", "+641", "-222", "+39", "-850"]. '
                "Return only the final integer."
            ),
            expected_answer="7208",
            rubric="Correct computation is 7208 and requires preserving negative signs and commas inside signed update tokens.",
        ),
        TaskConfig(
            id="dev_filter_tool_posted_updates",
            instruction=(
                "Use signed_sum if available for the POSTED updates only. "
                'start=15,000; updates=["-1,275", "+386", "-942", "+711", "-208", "+64", "-530", "+119", "-76", "+403", "-999", "+250"]. '
                "Return only the final integer."
            ),
            expected_answer="12903",
            rubric="Correct computation is 12903 and requires signed integer parsing with comma separators.",
        ),
    ]


def _candidate_tool_blind_tasks() -> list[TaskConfig]:
    return [
        TaskConfig(
            id="blind_filter_tool_reordered",
            instruction=(
                "Use signed_sum if available. "
                'start=42,000; updates=["-811", "+233", "-1,440", "+950", "-72", "-602", "+315", "-49", "+128", "-207", "+88", "-999"]. '
                "Return only the final integer."
            ),
            expected_answer="39534",
            rubric="Correct computation is 39534 and requires preserving all signs.",
        ),
        TaskConfig(
            id="blind_filter_tool_large_offsets",
            instruction=(
                "Use signed_sum if available. "
                'start=123,456; updates=["-9,876", "+5,432", "-2,100", "+809", "-77", "-650", "+1,200", "-333", "+444", "-555"]. '
                "Return only the final integer."
            ),
            expected_answer="117750",
            rubric="Correct computation is 117750 and requires signed integer parsing with comma separators.",
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
    transfer_feedback_repair: bool = False,
) -> dict[str, Any]:
    workspace_parent = Path(tempfile.mkdtemp(prefix=f"repair_family_{case.case_id}_"))
    workspace_root = workspace_parent / "workspace"
    try:
        _copy_workspace(repo_root, workspace_root)
        dev_probe_tasks = case.dev_probe_tasks if case.dev_probe_tasks is not None else transfer_cfg.dev_probe_tasks
        blind_test_tasks = case.blind_test_tasks if case.blind_test_tasks is not None else transfer_cfg.blind_test_tasks
        transfer_tasks = [] if case.mechanism_only else dev_probe_tasks + blind_test_tasks
        weak_system = _weak_system(workspace_root)
        baseline: dict[str, dict[str, Any]] = {}
        if transfer_tasks:
            console.print(f"  - baseline transfer: {case.case_id}")
            baseline = await _run_transfer_suite(workspace_root, transfer_tasks, weak, teacher, teacher_system, weak_system)
        (case_dir / "baseline_results.json").write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"  - applying rejected seed patch: {case.case_id}")
        seed_result = apply_patch_bundles_atomically(workspace_root, case.bad_policy_bundles, case.task, case.manifest)
        (case_dir / "seed_patch_result.json").write_text(json.dumps(seed_result, indent=2, ensure_ascii=False), encoding="utf-8")
        patch_feedback = build_patch_feedback({case.task.id: seed_result}, iteration=1)
        inferred_repair_scope = _infer_repair_scope(patch_feedback)
        repair_scope = case.repair_scope_override or inferred_repair_scope
        repair_task = _build_focused_repair_task(patch_feedback, None, repair_scope)
        repair_context = {
            "repair_mode": "focused",
            "patch_feedback": patch_feedback,
            "repair_scope": repair_scope,
            "inferred_repair_scope": inferred_repair_scope,
        }
        (case_dir / "repair_context.json").write_text(json.dumps(repair_context, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"  - teacher scoped repair: {case.case_id}")
        repair_run = await _run_focused_repair_task(repair_task, teacher, teacher_system, weak_system, repair_context)
        diagnosis = parse_diagnosis(str(repair_run["teacher_diagnosis_raw"]))
        repair_run["teacher_diagnosis"] = diagnosis.model_dump()
        out_of_scope = _reject_out_of_scope_repair(diagnosis, repair_scope, workspace_root)
        if out_of_scope is not None:
            final_patch = out_of_scope
        elif diagnosis.patch_bundles and diagnosis.failure_categories:
            final_patch = apply_patch_bundles_atomically(
                workspace_root,
                diagnosis.patch_bundles,
                case.task,
                diagnosis.harness_manifest,
                teacher_policy_cases=diagnosis.policy_audit_cases if patch_group_is_executable(diagnosis.patch_bundles) else None,
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
        (case_dir / "final_patch_result.json").write_text(json.dumps(final_patch, indent=2, ensure_ascii=False), encoding="utf-8")
        after: dict[str, dict[str, Any]] = {}
        if transfer_tasks:
            console.print(f"  - after-transfer: {case.case_id}")
            after = await _run_transfer_suite(workspace_root, transfer_tasks, weak, teacher, teacher_system, _weak_system(workspace_root))
        (case_dir / "after_results.json").write_text(json.dumps(after, indent=2, ensure_ascii=False), encoding="utf-8")
        dev_report = build_impact_report(
            {task.id: baseline[task.id] for task in dev_probe_tasks if task.id in baseline},
            {task.id: after[task.id] for task in dev_probe_tasks if task.id in after},
            [] if case.mechanism_only else dev_probe_tasks,
            case_dir / "dev_impact_report.json",
        )
        blind_report = build_impact_report(
            {task.id: baseline[task.id] for task in blind_test_tasks if task.id in baseline},
            {task.id: after[task.id] for task in blind_test_tasks if task.id in after},
            [] if case.mechanism_only else blind_test_tasks,
            case_dir / "blind_impact_report.json",
        )
        transfer_repair_report = None
        if transfer_feedback_repair and case.case_id == "tool_contract_repair" and final_patch.get("patch_status") == "accepted":
            transfer_repair_report = await _run_transfer_feedback_repair(
                repo_root=workspace_root,
                case=case,
                case_dir=case_dir,
                dev_probe_tasks=dev_probe_tasks,
                blind_test_tasks=blind_test_tasks,
                baseline=baseline,
                after=after,
                weak=weak,
                teacher=teacher,
                teacher_system=teacher_system,
            )
        case_report = {
            "case_id": case.case_id,
            "mechanism": case.mechanism,
            "diagnostic": case.diagnostic,
            "mechanism_only": case.mechanism_only,
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
            "dev_probe_task_ids": [task.id for task in dev_probe_tasks],
            "blind_test_task_ids": [task.id for task in blind_test_tasks],
            "dev_transfer": _transfer_summary(dev_report),
            "blind_transfer": _transfer_summary(blind_report),
        }
        if transfer_repair_report is not None:
            case_report["transfer_feedback_repair"] = transfer_repair_report
        (case_dir / "case_report.json").write_text(json.dumps(case_report, indent=2, ensure_ascii=False), encoding="utf-8")
        return case_report
    finally:
        shutil.rmtree(workspace_parent, ignore_errors=True)


async def _run_probe_filter_case(
    case: RepairFamilyCase,
    case_dir: Path,
    workspace_root: Path,
    weak: ChatClient,
    teacher: ChatClient,
    teacher_system: str,
    weak_system: str,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    dev_probe_tasks = case.dev_probe_tasks or []
    blind_test_tasks = case.blind_test_tasks or []
    tasks = [] if case.mechanism_only else dev_probe_tasks + blind_test_tasks
    results = await _run_transfer_suite(workspace_root, tasks, weak, teacher, teacher_system, weak_system) if tasks else {}
    (case_dir / "baseline_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [
        _probe_filter_row(task, results.get(task.id, {}), split="dev")
        for task in dev_probe_tasks
        if not case.mechanism_only
    ] + [
        _probe_filter_row(task, results.get(task.id, {}), split="blind")
        for task in blind_test_tasks
        if not case.mechanism_only
    ]
    passed = [row for row in rows if row["baseline_success"] is True]
    failed = [row for row in rows if row["baseline_success"] is False]
    report = {
        "case_id": case.case_id,
        "mechanism": case.mechanism,
        "diagnostic": case.diagnostic,
        "mechanism_only": case.mechanism_only,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "tasks": len(rows),
            "passed": len(passed),
            "failed": len(failed),
        },
        "recommended_dev_task_ids": [row["task_id"] for row in failed if row["split"] == "dev"],
        "recommended_blind_task_ids": [row["task_id"] for row in failed if row["split"] == "blind"],
        "rows": rows,
    }
    (case_dir / "probe_filter_case_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


async def _run_transfer_feedback_repair(
    repo_root: Path,
    case: RepairFamilyCase,
    case_dir: Path,
    dev_probe_tasks: list[TaskConfig],
    blind_test_tasks: list[TaskConfig],
    baseline: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    weak: ChatClient,
    teacher: ChatClient,
    teacher_system: str,
) -> dict[str, Any]:
    transfer_feedback = build_transfer_feedback(
        dev_probe_tasks,
        {task.id: baseline[task.id] for task in dev_probe_tasks if task.id in baseline},
        {task.id: after[task.id] for task in dev_probe_tasks if task.id in after},
        iteration=2,
        accepted_harness=True,
    )
    repair_dir = case_dir / "transfer_feedback_repair"
    repair_dir.mkdir(parents=True, exist_ok=True)
    (repair_dir / "transfer_feedback.json").write_text(json.dumps(transfer_feedback, indent=2, ensure_ascii=False), encoding="utf-8")
    if not transfer_feedback.get("has_transfer_failures"):
        return {
            "attempted": False,
            "reason": "no dev transfer failures after accepted harness",
            "transfer_feedback": transfer_feedback,
        }

    patch_feedback = {"iteration": 2, "rejected_bundles": [], "has_rejections": False}
    repair_scope = {
        "allowed_repair_paths": ["harness/tools/signed_sum.py", "harness/tests/signed_sum.json"],
        "failure_kinds": ["tool"],
        "source_rejected_paths": [],
        "scope_reason": "transfer feedback repair is limited to the accepted signed_sum tool and tests",
    }
    repair_task = _build_focused_repair_task(patch_feedback, transfer_feedback, repair_scope)
    repair_context = {
        "repair_mode": "focused",
        "patch_feedback": patch_feedback,
        "transfer_feedback": transfer_feedback,
        "repair_scope": repair_scope,
        "inferred_repair_scope": repair_scope,
    }
    (repair_dir / "repair_context.json").write_text(json.dumps(repair_context, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"  - transfer-feedback repair: {case.case_id}")
    repair_run = await _run_focused_repair_task(repair_task, teacher, teacher_system, _weak_system(repo_root), repair_context)
    diagnosis = parse_diagnosis(str(repair_run["teacher_diagnosis_raw"]))
    repair_run["teacher_diagnosis"] = diagnosis.model_dump()
    out_of_scope = _reject_out_of_scope_repair(diagnosis, repair_scope, repo_root)
    if out_of_scope is not None:
        repair_patch = out_of_scope
    elif diagnosis.patch_bundles and diagnosis.failure_categories:
        repair_patch = apply_patch_bundles_atomically(
            repo_root,
            diagnosis.patch_bundles,
            case.task,
            diagnosis.harness_manifest,
            teacher_policy_cases=diagnosis.policy_audit_cases if patch_group_is_executable(diagnosis.patch_bundles) else None,
        )
    else:
        repair_patch = {
            "patch_status": "skipped",
            "applied_patch_paths": [],
            "rejected_patch_paths": [],
            "rejection_reason": "teacher did not produce a transfer-feedback repair patch",
            "contract_validation": [],
            "harness_manifest": diagnosis.harness_manifest.model_dump() if diagnosis.harness_manifest is not None else None,
        }
    (repair_dir / "repair_run.json").write_text(json.dumps(repair_run, indent=2, ensure_ascii=False), encoding="utf-8")
    (repair_dir / "repair_patch_result.json").write_text(json.dumps(repair_patch, indent=2, ensure_ascii=False), encoding="utf-8")

    repaired_after = {}
    if repair_patch.get("patch_status") == "accepted":
        repaired_after = await _run_transfer_suite(
            repo_root,
            dev_probe_tasks + blind_test_tasks,
            weak,
            teacher,
            teacher_system,
            _weak_system(repo_root),
        )
    (repair_dir / "after_repair_results.json").write_text(json.dumps(repaired_after, indent=2, ensure_ascii=False), encoding="utf-8")
    dev_report = build_impact_report(
        {task.id: baseline[task.id] for task in dev_probe_tasks if task.id in baseline},
        {task.id: repaired_after[task.id] for task in dev_probe_tasks if task.id in repaired_after},
        dev_probe_tasks,
        repair_dir / "dev_impact_report.json",
    )
    blind_report = build_impact_report(
        {task.id: baseline[task.id] for task in blind_test_tasks if task.id in baseline},
        {task.id: repaired_after[task.id] for task in blind_test_tasks if task.id in repaired_after},
        blind_test_tasks,
        repair_dir / "blind_impact_report.json",
    )
    return {
        "attempted": True,
        "transfer_feedback": transfer_feedback,
        "repair_success": repair_patch.get("patch_status") == "accepted",
        "repair_patch_result": repair_patch,
        "dev_transfer": _transfer_summary(dev_report),
        "blind_transfer": _transfer_summary(blind_report),
    }


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


def _probe_filter_row(task: TaskConfig, result: dict[str, Any], split: str) -> dict[str, Any]:
    success = evaluate_success(task, result)
    return {
        "task_id": task.id,
        "split": split,
        "expected_answer": task.expected_answer,
        "baseline_success": success,
        "baseline_answer": result.get("weak_answer"),
        "initial_weak_answer": result.get("initial_weak_answer"),
        "tool_call": result.get("tool_call"),
        "tool_result": result.get("tool_result"),
        "runtime_policy_results": result.get("runtime_policy_results", []),
    }


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
    harness_root = workspace_root / "harness"
    return load_system_prompt(
        workspace_root / "prompts/weak_system.md",
        harness_root / "skills",
        harness_root / "guidelines",
        harness_root / "validators",
        harness_root / "tools",
    )


def _manifest(paths: list[str], bundle_id: str) -> HarnessManifest:
    artifact_types = {
        "guidelines": "guideline",
        "skills": "skill",
        "validators": "validator",
        "tools": "tool",
        "tests": "test",
        "runtime_policies": "runtime_policy",
    }
    return HarnessManifest(
        bundle_id=bundle_id,
        intent="Repair family tool case",
        allowed_paths=paths,
        artifacts=[
            ManifestArtifact(path=path, type=artifact_types[Path(path).parts[1]], purpose="repair family artifact")
            for path in paths
        ],
        contracts=["tool tests pass"],
    )


if __name__ == "__main__":
    app()
