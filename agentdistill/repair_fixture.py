from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import typer

from agentdistill.config import TaskConfig
from agentdistill.diagnosis import PatchBundle
from agentdistill.manifest import HarnessManifest
from agentdistill.patches import apply_patch_bundles_atomically
from agentdistill.repair_efficiency import build_repair_efficiency_report


app = typer.Typer(add_completion=False)


@app.command()
def main(
    output_dir: Path = typer.Option(Path("outputs/repair_mechanism_fixture"), "--output-dir", "-o"),
) -> None:
    report = run_repair_fixture(output_dir)
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


def run_repair_fixture(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [
        _write_fixture_run(output_dir / "fixture_full_train", mode="full_train"),
        _write_fixture_run(output_dir / "fixture_focused_only", mode="focused_only"),
        _write_fixture_run(output_dir / "fixture_scoped_inner", mode="scoped_inner"),
    ]
    report = build_repair_efficiency_report(run_dirs)
    (output_dir / "repair_efficiency_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _write_fixture_run(run_dir: Path, mode: str) -> Path:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    _make_harness_dirs(run_dir)

    task = TaskConfig(
        id="fixture_train",
        instruction="Use the signed updates: start=100, updates=[+25, -7]. Return the final count.",
        expected_answer="118",
    )
    rejected_result = apply_patch_bundles_atomically(
        run_dir,
        _bad_policy_bundles(),
        task,
        _manifest(["harness/runtime_policies/force_fixture.py", "harness/tests/force_fixture.json"]),
    )
    rejected_result.update({"task_id": task.id, "created_at": "2026-01-01T00:00:00+00:00"})
    inner_attempts: list[dict[str, Any]] = []
    if mode == "scoped_inner":
        inner = apply_patch_bundles_atomically(
            run_dir,
            _good_policy_bundles(),
            task,
            _manifest(["harness/runtime_policies/force_fixture.py", "harness/tests/force_fixture.json"]),
        )
        inner.update(
            {
                "task_id": "focused_repair",
                "created_at": "2026-01-01T00:00:01+00:00",
                "focused_repair": True,
                "inner_repair_attempt": 1,
                "context_repair_scope": {
                    "allowed_repair_paths": [
                        "harness/runtime_policies/force_fixture.py",
                        "harness/tests/force_fixture.json",
                    ],
                    "failure_kinds": ["runtime_policy"],
                },
            }
        )
        inner_attempts.append(inner)
        phase_dir = run_dir / "evolve_train_iter_01"
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "fixture_train.inner_repair_1.json").write_text(
            json.dumps(inner, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    row = _summary_row(rejected_result, mode, inner_attempts)
    _write_run_files(run_dir, row)
    return run_dir


def _summary_row(result: dict[str, Any], mode: str, inner_attempts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "iteration": 1,
        "phase_kind": "focused_repair" if mode == "focused_only" else "full_train",
        "task_id": "fixture_train",
        "created_at": result.get("created_at"),
        "applied_patch_paths": result.get("applied_patch_paths", []),
        "rejected_patch_paths": result.get("rejected_patch_paths", []),
        "patch_status": result.get("patch_status"),
        "contract_validation": result.get("contract_validation"),
        "harness_manifest": result.get("harness_manifest"),
        "context_repair_scope": {
            "allowed_repair_paths": ["harness/runtime_policies/force_fixture.py", "harness/tests/force_fixture.json"],
            "failure_kinds": ["runtime_policy"],
        }
        if mode == "scoped_inner"
        else None,
        "inner_repair_attempts": inner_attempts,
        "rejection_reason": result.get("rejection_reason"),
        "failure_categories": ["runtime_policy"],
    }


def _write_run_files(run_dir: Path, row: dict[str, Any]) -> None:
    phase_dir = run_dir / "evolve_train_iter_01"
    phase_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "task_id": row["task_id"],
        "created_at": row["created_at"],
        "patch_status": row["patch_status"],
        "applied_patch_paths": row["applied_patch_paths"],
        "rejected_patch_paths": row["rejected_patch_paths"],
        "contract_validation": row["contract_validation"],
        "harness_manifest": row["harness_manifest"],
        "context_repair_scope": row["context_repair_scope"],
        "inner_repair_attempts": row["inner_repair_attempts"],
        "rejection_reason": row["rejection_reason"],
        "teacher_diagnosis": {"failure_categories": row["failure_categories"]},
    }
    (phase_dir / "fixture_train.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (phase_dir / "summary.json").write_text(json.dumps([result], indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "train_summary.json").write_text(json.dumps([row], indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "dev_impact_report.json").write_text(json.dumps([], indent=2), encoding="utf-8")
    (run_dir / "blind_impact_report.json").write_text(json.dumps([], indent=2), encoding="utf-8")
    (run_dir / "harness_files_after.json").write_text(json.dumps(_harness_files(run_dir), indent=2), encoding="utf-8")


def _bad_policy_bundles() -> list[PatchBundle]:
    return [
        PatchBundle(
            target_path="harness/runtime_policies/force_fixture.py",
            content="""
def evaluate(input: dict) -> dict:
    return {"requires_tool": True, "tool_name": "missing_fixture_tool", "tool_input": {}}
""".strip(),
        ),
        PatchBundle(
            target_path="harness/tests/force_fixture.json",
            content="""
{
  "policy": "force_fixture",
  "cases": [
    {
      "input": {
        "task_instruction": "Use the signed updates: start=100, updates=[+25, -7]. Return the final count.",
        "available_tools": [],
        "expected_answer": "118"
      },
      "expected": {"requires_tool": true, "tool_name": "missing_fixture_tool"},
      "expected_tool_result": {"ok": true, "result": 118}
    }
  ]
}
""".strip(),
        ),
    ]


def _good_policy_bundles() -> list[PatchBundle]:
    return [
        PatchBundle(
            target_path="harness/runtime_policies/force_fixture.py",
            content="""
def evaluate(input: dict) -> dict:
    return {"requires_tool": False, "reason": "fixture repair disables invalid missing tool route"}
""".strip(),
        ),
        PatchBundle(
            target_path="harness/tests/force_fixture.json",
            content="""
{
  "policy": "force_fixture",
  "cases": [
    {
      "input": {
        "task_instruction": "Use the signed updates: start=100, updates=[+25, -7]. Return the final count.",
        "available_tools": [],
        "expected_answer": "118"
      },
      "expected": {"requires_tool": false}
    }
  ]
}
""".strip(),
        ),
    ]


def _manifest(paths: list[str]) -> HarnessManifest:
    artifacts = []
    for path in paths:
        artifacts.append(
            {
                "path": path,
                "type": "runtime_policy" if "/runtime_policies/" in path else "test",
                "purpose": "deterministic repair fixture artifact",
            }
        )
    return HarnessManifest(
        bundle_id="repair_fixture",
        intent="deterministically exercise rejected patch repair mechanics",
        allowed_paths=paths,
        artifacts=artifacts,
        contracts=["fixture contracts pass"],
    )


def _make_harness_dirs(root: Path) -> None:
    for subdir in ["guidelines", "skills", "validators", "tools", "runtime_policies", "tests"]:
        (root / "harness" / subdir).mkdir(parents=True, exist_ok=True)


def _harness_files(root: Path) -> list[str]:
    files = []
    harness = root / "harness"
    for path in sorted(harness.rglob("*")):
        if path.is_file():
            files.append(str(path.relative_to(root)))
    return files


if __name__ == "__main__":
    app()
