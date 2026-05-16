from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime


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
    blind_impact_rows: list[dict[str, Any]] | None = None,
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
    manifest_rows = [row for row in train_summary if isinstance(row.get("harness_manifest"), dict)]
    code_bundle_rows = [row for row in manifest_rows if _manifest_has_code_artifact(row["harness_manifest"])]
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
            "manifest_bundles": len(manifest_rows),
            "code_manifest_bundles": len(code_bundle_rows),
            "accepted_code_manifest_bundles": sum(
                1
                for row in accepted
                if isinstance(row.get("harness_manifest"), dict) and _manifest_has_code_artifact(row["harness_manifest"])
            ),
            "contract_failures": _count_contract_failures(train_summary),
        },
        "transfer": _build_transfer_metrics(impact_rows),
        "dev_transfer": _build_transfer_metrics(impact_rows),
        "blind_transfer": _build_transfer_metrics(blind_impact_rows or impact_rows),
        "repair_efficiency": _build_repair_efficiency_metrics(train_summary, impact_rows, blind_impact_rows or impact_rows),
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


def _manifest_has_code_artifact(manifest: dict[str, Any]) -> bool:
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        return False
    return any(
        isinstance(artifact, dict) and artifact.get("type") in {"tool", "runtime_policy", "test"}
        for artifact in artifacts
    )


def _build_transfer_metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "heldout_tasks": len(rows),
        "before_success": sum(1 for row in rows if row.get("before_success") is True),
        "after_success": sum(1 for row in rows if row.get("after_success") is True),
        "improved": sum(1 for row in rows if row.get("improved") is True),
        "regressed": sum(1 for row in rows if row.get("regressed") is True),
    }


def _build_repair_efficiency_metrics(
    train_summary: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    blind_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    patch_rows = [row for row in train_summary if row.get("patch_status") in {"accepted", "rejected"}]
    accepted = [row for row in patch_rows if row.get("patch_status") == "accepted"]
    rejected = [row for row in patch_rows if row.get("patch_status") == "rejected"]
    inner_attempts = [
        attempt
        for row in train_summary
        for attempt in row.get("inner_repair_attempts", [])
        if isinstance(attempt, dict)
    ]
    inner_accepted = [attempt for attempt in inner_attempts if attempt.get("patch_status") == "accepted"]
    scoped_attempts = [attempt for attempt in inner_attempts if isinstance(attempt.get("context_repair_scope"), dict)]
    scoped_accepted = [attempt for attempt in scoped_attempts if attempt.get("patch_status") == "accepted"]
    out_of_scope_rejections = [
        attempt
        for attempt in inner_attempts
        if attempt.get("rejection_reason") == "inner repair patch targets outside allowed repair scope"
    ]
    touched_paths = [_normalize_harness_path(path) for row in patch_rows for path in _row_patch_paths(row)]
    touched_paths = [path for path in touched_paths if path is not None]
    return {
        "patch_attempts": len(patch_rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_rate": _safe_ratio(len(accepted), len(patch_rows)),
        "repair_success": bool(accepted or inner_accepted),
        "repair_success_via": _repair_success_via(accepted, inner_accepted, scoped_accepted),
        "inner_repair_attempts": len(inner_attempts),
        "inner_repair_accepted": len(inner_accepted),
        "inner_repair_rejected": sum(1 for attempt in inner_attempts if attempt.get("patch_status") == "rejected"),
        "scoped_inner_repair_attempts": len(scoped_attempts),
        "scoped_inner_repair_accepted": len(scoped_accepted),
        "scoped_inner_repair_success": bool(scoped_accepted),
        "out_of_scope_rejections": len(out_of_scope_rejections),
        "total_patch_paths": len(touched_paths),
        "unique_patch_paths": len(set(touched_paths)),
        "avg_paths_per_patch_attempt": _safe_ratio(len(touched_paths), len(patch_rows)),
        "path_type_counts": _count_path_types(touched_paths),
        "cost_proxies": _build_cost_proxies(train_summary, inner_attempts),
        "dev": _build_transfer_metrics(dev_rows),
        "blind": _build_transfer_metrics(blind_rows),
    }


def _row_patch_paths(row: dict[str, Any]) -> list[str]:
    paths = []
    for key in ("applied_patch_paths", "rejected_patch_paths"):
        value = row.get(key, [])
        if isinstance(value, list):
            paths.extend(path for path in value if isinstance(path, str))
    return paths


def _normalize_harness_path(path: str) -> str | None:
    parts = Path(path).parts
    if "harness" not in parts:
        return None
    index = parts.index("harness")
    return "/".join(parts[index:])


def _safe_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _repair_success_via(
    accepted: list[dict[str, Any]],
    inner_accepted: list[dict[str, Any]],
    scoped_accepted: list[dict[str, Any]],
) -> str:
    if accepted:
        return "outer_patch"
    if scoped_accepted:
        return "scoped_inner_repair"
    if inner_accepted:
        return "inner_repair"
    return "none"


def _build_cost_proxies(train_summary: list[dict[str, Any]], inner_attempts: list[dict[str, Any]]) -> dict[str, Any]:
    teacher_calls = sum(1 for row in train_summary if row.get("patch_status") in {"accepted", "rejected", "skipped"})
    teacher_calls += len(inner_attempts)
    focused_rows = [row for row in train_summary if row.get("phase_kind") == "focused_repair"]
    focused_inner = [attempt for attempt in inner_attempts if attempt.get("focused_repair") is True]
    created_at_values = [
        value
        for row in train_summary
        for value in [row.get("created_at")]
        if isinstance(value, str)
    ]
    span_seconds = _created_at_span_seconds(created_at_values)
    return {
        "teacher_call_proxy": teacher_calls,
        "weak_call_proxy": max(0, len(train_summary) - len(focused_rows)),
        "focused_repair_weak_calls_skipped": len(focused_rows) + len(focused_inner),
        "train_created_at_span_seconds": span_seconds,
    }


def _created_at_span_seconds(values: list[str]) -> float | None:
    timestamps = []
    for value in values:
        try:
            timestamps.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            continue
    if len(timestamps) < 2:
        return None
    return round((max(timestamps) - min(timestamps)).total_seconds(), 3)
