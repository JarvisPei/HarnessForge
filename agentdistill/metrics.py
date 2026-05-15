from __future__ import annotations

from pathlib import Path
from typing import Any


HARNESS_BUCKETS = {
    "guidelines": "guideline",
    "skills": "skill",
    "validators": "validator",
    "tools": "tool",
    "runtime_policies": "runtime_policy",
    "tests": "test",
}


def build_benchmark_metrics(
    train_summary: list[dict[str, Any]],
    impact_rows: list[dict[str, Any]],
    harness_files_after: list[str],
) -> dict[str, Any]:
    accepted = [row for row in train_summary if row.get("patch_status") == "accepted"]
    rejected = [row for row in train_summary if row.get("patch_status") == "rejected"]
    applied_paths = [
        path
        for row in train_summary
        for path in row.get("applied_patch_paths", [])
        if isinstance(path, str)
    ]
    rejected_paths = [
        path
        for row in train_summary
        for path in row.get("rejected_patch_paths", [])
        if isinstance(path, str)
    ]
    type_counts = _count_path_types(applied_paths)
    harness_type_counts = _count_path_types(harness_files_after)
    return {
        "patches": {
            "train_steps": len(train_summary),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "applied_paths": len(applied_paths),
            "rejected_paths": len(rejected_paths),
            "applied_type_counts": type_counts,
            "accepted_with_tool": _count_rows_with_type(accepted, "tool"),
            "accepted_with_test": _count_rows_with_type(accepted, "test"),
            "accepted_with_runtime_policy": _count_rows_with_type(accepted, "runtime_policy"),
            "accepted_tool_test_policy_bundles": sum(
                1
                for row in accepted
                if _row_has_type(row, "tool") and _row_has_type(row, "test") and _row_has_type(row, "runtime_policy")
            ),
            "contract_failures": _count_contract_failures(train_summary),
        },
        "transfer": {
            "heldout_tasks": len(impact_rows),
            "before_success": sum(1 for row in impact_rows if row.get("before_success") is True),
            "after_success": sum(1 for row in impact_rows if row.get("after_success") is True),
            "improved": sum(1 for row in impact_rows if row.get("improved") is True),
            "regressed": sum(1 for row in impact_rows if row.get("regressed") is True),
        },
        "harness_after": {
            "files": len(harness_files_after),
            "type_counts": harness_type_counts,
        },
    }


def _count_path_types(paths: list[str]) -> dict[str, int]:
    counts = {bucket: 0 for bucket in HARNESS_BUCKETS.values()}
    for path in paths:
        path_type = _path_type(path)
        if path_type is not None:
            counts[path_type] += 1
    return counts


def _path_type(path: str) -> str | None:
    parts = Path(path).parts
    if "harness" not in parts:
        return None
    index = parts.index("harness")
    if index + 1 >= len(parts):
        return None
    return HARNESS_BUCKETS.get(parts[index + 1])


def _row_has_type(row: dict[str, Any], path_type: str) -> bool:
    return any(_path_type(path) == path_type for path in row.get("applied_patch_paths", []) if isinstance(path, str))


def _count_rows_with_type(rows: list[dict[str, Any]], path_type: str) -> int:
    return sum(1 for row in rows if _row_has_type(row, path_type))


def _count_contract_failures(train_summary: list[dict[str, Any]]) -> int:
    failures = 0
    for row in train_summary:
        validation = row.get("contract_validation", [])
        if isinstance(validation, dict):
            validation = [validation]
        if isinstance(validation, list):
            failures += sum(1 for item in validation if isinstance(item, dict) and item.get("ok") is not True)
    return failures
