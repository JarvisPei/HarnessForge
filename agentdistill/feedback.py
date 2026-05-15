from __future__ import annotations

from typing import Any


MAX_FAILURES_PER_RESULT = 3


def build_patch_feedback(train_results: dict[str, dict[str, Any]], iteration: int) -> dict[str, Any]:
    rejected = []
    for task_id, result in train_results.items():
        if result.get("patch_status") != "rejected":
            continue
        rejected.append(_summarize_rejection(task_id, result))
    return {
        "iteration": iteration,
        "rejected_bundles": rejected,
        "has_rejections": bool(rejected),
    }


def merge_benchmark_context(
    transfer_context: dict[str, Any],
    patch_feedback: dict[str, Any] | None,
) -> dict[str, Any]:
    if not patch_feedback or not patch_feedback.get("has_rejections"):
        return transfer_context
    return {**transfer_context, "patch_feedback": patch_feedback}


def _summarize_rejection(task_id: str, result: dict[str, Any]) -> dict[str, Any]:
    manifest = result.get("harness_manifest")
    bundle_id = manifest.get("bundle_id") if isinstance(manifest, dict) else None
    return {
        "task_id": task_id,
        "bundle_id": bundle_id,
        "rejection_reason": result.get("rejection_reason"),
        "rejected_patch_paths": result.get("rejected_patch_paths", []),
        "failed_contracts": _summarize_failed_contracts(result.get("contract_validation", [])),
    }


def _summarize_failed_contracts(validation: Any) -> list[dict[str, Any]]:
    if isinstance(validation, dict):
        validation = [validation]
    if not isinstance(validation, list):
        return []
    failed = []
    for item in validation:
        if not isinstance(item, dict) or item.get("ok") is True:
            continue
        summary = {
            "path": item.get("path"),
            "reason": item.get("reason"),
            "policy": item.get("policy"),
            "tool": item.get("tool"),
            "expected_answer": item.get("expected_answer"),
        }
        if "policy_result" in item:
            summary["policy_result"] = _compact(item["policy_result"])
        if "tool_result" in item:
            summary["tool_result"] = _compact(item["tool_result"])
        if "failures" in item:
            summary["failures"] = _compact(item["failures"], max_list=MAX_FAILURES_PER_RESULT)
        failed.append({key: value for key, value in summary.items() if value is not None})
    return failed


def _compact(value: Any, max_list: int = 6) -> Any:
    if isinstance(value, dict):
        return {str(key): _compact(item, max_list=max_list) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact(item, max_list=max_list) for item in value[:max_list]]
    return value
