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


def build_transfer_feedback(
    tasks: list[Any],
    baseline_results: dict[str, dict[str, Any]],
    probe_results: dict[str, dict[str, Any]],
    iteration: int,
    accepted_harness: bool,
    previous_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unresolved_by_id = _previous_unresolved_by_id(previous_feedback)
    if not accepted_harness:
        return _transfer_feedback_result(iteration, unresolved_by_id)

    current_failed_by_id = {}
    current_success_ids = set()
    for task in tasks:
        before = baseline_results.get(task.id, {})
        after = probe_results.get(task.id, {})
        before_success = _result_success(task.expected_answer, before)
        after_success = _result_success(task.expected_answer, after)
        if after_success:
            current_success_ids.add(task.id)
            continue
        previous = unresolved_by_id.get(task.id, {})
        first_seen = previous.get("first_seen_iteration", iteration)
        current_failed_by_id[task.id] = {
            "task_id": task.id,
            "task_instruction": task.instruction,
            "expected_answer": task.expected_answer,
            "rubric": task.rubric,
            "failure_mode": _classify_transfer_failure(before, after),
            "recommended_repair_target": _recommend_transfer_repair_target(before, after),
            "before_success": before_success,
            "after_success": after_success,
            "regressed": before_success and not after_success,
            "before_answer": before.get("weak_answer"),
            "after_answer": after.get("weak_answer"),
            "before_tool_call": _compact(before.get("tool_call")),
            "after_tool_call": _compact(after.get("tool_call")),
            "before_tool_result": _compact(before.get("tool_result")),
            "after_tool_result": _compact(after.get("tool_result")),
            "after_runtime_policy_results": _compact(after.get("runtime_policy_results", [])),
            "first_seen_iteration": first_seen,
            "last_seen_iteration": iteration,
        }

    for task_id in current_success_ids:
        unresolved_by_id.pop(task_id, None)
    unresolved_by_id.update(current_failed_by_id)
    return _transfer_feedback_result(iteration, unresolved_by_id)


def merge_benchmark_context(
    transfer_context: dict[str, Any],
    patch_feedback: dict[str, Any] | None,
    transfer_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = dict(transfer_context)
    if not patch_feedback or not patch_feedback.get("has_rejections"):
        pass
    else:
        context["patch_feedback"] = patch_feedback
    if transfer_feedback and transfer_feedback.get("has_transfer_failures"):
        context["transfer_feedback"] = transfer_feedback
    return context


def _result_success(expected_answer: str | None, result: dict[str, Any]) -> bool:
    if not expected_answer:
        return bool(result.get("weak_answer"))
    expected_numbers = _numbers(expected_answer)
    answer_numbers = _numbers(str(result.get("weak_answer", "")))
    if expected_numbers and not all(num in answer_numbers for num in expected_numbers):
        return False
    diagnosis = result.get("teacher_diagnosis", {})
    categories = diagnosis.get("failure_categories", []) if isinstance(diagnosis, dict) else []
    return not bool(categories)


def _numbers(text: str) -> list[str]:
    import re

    return [match.replace(",", "") for match in re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)]


def _previous_unresolved_by_id(feedback: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not feedback or not feedback.get("has_transfer_failures"):
        return {}
    failed_tasks = feedback.get("failed_tasks", [])
    if not isinstance(failed_tasks, list):
        return {}
    return {
        str(item["task_id"]): dict(item)
        for item in failed_tasks
        if isinstance(item, dict) and item.get("task_id")
    }


def _transfer_feedback_result(iteration: int, unresolved_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failed_tasks = list(unresolved_by_id.values())
    return {
        "iteration": iteration,
        "has_transfer_failures": bool(failed_tasks),
        "failed_tasks": failed_tasks,
    }


def _classify_transfer_failure(before: dict[str, Any], after: dict[str, Any]) -> str:
    after_runtime = after.get("runtime_policy_results", [])
    if isinstance(after_runtime, list):
        forced = [item for item in after_runtime if isinstance(item, dict) and item.get("requires_tool")]
        if forced:
            after_tool = after.get("tool_result")
            if isinstance(after_tool, dict):
                if after_tool.get("ok") is False:
                    return "tool_failure"
                return "finalization_failure"
            return "policy_or_routing_failure"
    before_tool = before.get("tool_result")
    after_tool = after.get("tool_result")
    if isinstance(before_tool, dict) and before_tool.get("ok") is True and isinstance(after_tool, dict) and after_tool.get("ok") is False:
        return "tool_failure"
    if after.get("tool_call") is None:
        return "policy_or_routing_failure"
    return "finalization_failure"


def _recommend_transfer_repair_target(before: dict[str, Any], after: dict[str, Any]) -> str:
    failure_mode = _classify_transfer_failure(before, after)
    if failure_mode == "tool_failure":
        return "tool"
    if failure_mode == "policy_or_routing_failure":
        return "runtime_policy"
    return "finalization"


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
