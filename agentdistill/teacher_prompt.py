from __future__ import annotations

import json
from typing import Any


def build_teacher_messages(
    teacher_system: str,
    *,
    task_id: str,
    task_instruction: str,
    expected_answer: str | None,
    rubric: str | None,
    weak_system_prompt: str,
    weak_answer: str,
    initial_weak_answer: str,
    tool_call: dict[str, object] | None,
    tool_result: dict[str, object] | None,
    runtime_policy_results: list[dict[str, object]],
    benchmark_context: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    payload = build_teacher_payload(
        task_id=task_id,
        task_instruction=task_instruction,
        expected_answer=expected_answer,
        rubric=rubric,
        weak_system_prompt=weak_system_prompt,
        weak_answer=weak_answer,
        initial_weak_answer=initial_weak_answer,
        tool_call=tool_call,
        tool_result=tool_result,
        runtime_policy_results=runtime_policy_results,
        benchmark_context=benchmark_context,
    )
    return [
        {"role": "system", "content": teacher_system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_teacher_payload(
    *,
    task_id: str,
    task_instruction: str,
    expected_answer: str | None,
    rubric: str | None,
    weak_system_prompt: str,
    weak_answer: str,
    initial_weak_answer: str,
    tool_call: dict[str, object] | None,
    tool_result: dict[str, object] | None,
    runtime_policy_results: list[dict[str, object]],
    benchmark_context: dict[str, object] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "task_instruction": task_instruction,
        "expected_answer": expected_answer,
        "rubric": rubric,
        "weak_system_prompt": weak_system_prompt,
        "weak_answer": weak_answer,
        "initial_weak_answer": initial_weak_answer,
        "tool_call": tool_call,
        "tool_result": tool_result,
        "runtime_policy_results": runtime_policy_results,
    }
    enriched_context = enrich_benchmark_context(benchmark_context)
    if enriched_context is not None:
        payload["benchmark_context"] = enriched_context
    return payload


def enrich_benchmark_context(benchmark_context: dict[str, object] | None) -> dict[str, object] | None:
    if not benchmark_context:
        return None
    context = dict(benchmark_context)
    repair_plan = _derive_repair_plan(context)
    if repair_plan is not None:
        context["repair_plan"] = repair_plan
    artifact_boundaries = _derive_artifact_boundaries(context, repair_plan)
    if artifact_boundaries is not None:
        context["artifact_boundaries"] = artifact_boundaries
    return context


def _derive_repair_plan(context: dict[str, object]) -> dict[str, object] | None:
    repair_plan = context.get("repair_plan")
    if isinstance(repair_plan, dict):
        return dict(repair_plan)

    transfer_feedback = context.get("transfer_feedback")
    if isinstance(transfer_feedback, dict):
        transfer_plan = transfer_feedback.get("repair_plan")
        if isinstance(transfer_plan, dict):
            return dict(transfer_plan)
        failed_tasks = transfer_feedback.get("failed_tasks")
        if isinstance(failed_tasks, list):
            for failed_task in failed_tasks:
                if not isinstance(failed_task, dict):
                    continue
                task_plan = failed_task.get("repair_plan")
                if isinstance(task_plan, dict):
                    return dict(task_plan)

    repair_scope = context.get("repair_scope")
    if isinstance(repair_scope, dict):
        allowed_artifact_types = _allowed_artifact_types_from_scope(repair_scope)
        if allowed_artifact_types:
            return {
                "primary_axis": "artifact_scope",
                "allowed_artifact_types": allowed_artifact_types,
                "required_regression_test": "regression test covering the rejected contract on the allowed repair paths",
            }

    patch_feedback = context.get("patch_feedback")
    if isinstance(patch_feedback, dict) and patch_feedback.get("has_rejections"):
        return {
            "primary_axis": "artifact_scope",
            "allowed_artifact_types": ["tool", "runtime_policy", "test", "guideline", "validator", "skill"],
            "required_regression_test": "regression test covering the rejected contract",
        }

    return None


def _derive_artifact_boundaries(
    context: dict[str, object],
    repair_plan: dict[str, object] | None,
) -> dict[str, object] | None:
    repair_scope = context.get("repair_scope")
    if not isinstance(repair_scope, dict) and repair_plan is None:
        return None

    boundaries: dict[str, object] = {}
    if isinstance(repair_scope, dict):
        for key in ["allowed_repair_paths", "failure_kinds", "source_rejected_paths", "scope_reason"]:
            value = repair_scope.get(key)
            if value is not None:
                boundaries[key] = value
    if isinstance(repair_plan, dict):
        if repair_plan.get("allowed_artifact_types") is not None:
            boundaries["allowed_artifact_types"] = repair_plan["allowed_artifact_types"]
        if repair_plan.get("required_regression_test") is not None:
            boundaries["required_regression_test"] = repair_plan["required_regression_test"]
    return boundaries or None


def _allowed_artifact_types_from_scope(repair_scope: dict[str, object]) -> list[str]:
    allowed_paths = repair_scope.get("allowed_repair_paths", [])
    if not isinstance(allowed_paths, list):
        return []
    artifact_types: list[str] = []
    for path in allowed_paths:
        if not isinstance(path, str):
            continue
        parts = path.split("/")
        if len(parts) != 3 or parts[0] != "harness":
            continue
        artifact_type = parts[1]
        if artifact_type == "tools":
            artifact_types.extend(["tool", "test"])
        elif artifact_type == "runtime_policies":
            artifact_types.extend(["runtime_policy", "test"])
        elif artifact_type == "tests":
            artifact_types.append("test")
        elif artifact_type == "guidelines":
            artifact_types.append("guideline")
        elif artifact_type == "skills":
            artifact_types.append("skill")
        elif artifact_type == "validators":
            artifact_types.append("validator")
    return sorted(set(artifact_types))
