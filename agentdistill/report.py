from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agentdistill.config import TaskConfig


def evaluate_success(task: TaskConfig, result: dict[str, Any]) -> bool:
    answer = str(result.get("weak_answer", ""))
    diagnosis = result.get("teacher_diagnosis", {})
    categories = diagnosis.get("failure_categories", []) if isinstance(diagnosis, dict) else []
    if categories:
        return False
    if task.expected_answer:
        json_success = _json_answer_matches(task.expected_answer, answer)
        if json_success is not None:
            return json_success
        expected_numbers = _numbers(task.expected_answer)
        answer_numbers = _numbers(answer)
        if expected_numbers and not all(num in answer_numbers for num in expected_numbers):
            return False
    return True


def build_impact_report(
    baseline: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    tasks: list[TaskConfig],
    output_path: Path,
) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        before_result = baseline.get(task.id)
        after_result = after.get(task.id)
        before_success = evaluate_success(task, before_result or {}) if before_result else False
        after_success = evaluate_success(task, after_result or {}) if after_result else False
        rows.append(
            {
                "task_id": task.id,
                "expected_answer": task.expected_answer,
                "before_success": before_success,
                "after_success": after_success,
                "improved": (not before_success) and after_success,
                "regressed": before_success and (not after_success),
                "before_answer": (before_result or {}).get("weak_answer"),
                "after_answer": (after_result or {}).get("weak_answer"),
                "before_tool_call": (before_result or {}).get("tool_call"),
                "after_tool_call": (after_result or {}).get("tool_call"),
                "before_tool_result": (before_result or {}).get("tool_result"),
                "after_tool_result": (after_result or {}).get("tool_result"),
                "before_runtime_policy_results": (before_result or {}).get("runtime_policy_results", []),
                "after_runtime_policy_results": (after_result or {}).get("runtime_policy_results", []),
                "after_runtime_policy_fired": _runtime_policy_fired(
                    (after_result or {}).get("runtime_policy_results", [])
                ),
                "after_runtime_effect": _runtime_effect(after_result or {}),
                "before_failures": ((before_result or {}).get("teacher_diagnosis") or {}).get("failure_categories", []),
                "after_failures": ((after_result or {}).get("teacher_diagnosis") or {}).get("failure_categories", []),
                "before_patch_status": (before_result or {}).get("patch_status"),
                "after_patch_status": (after_result or {}).get("patch_status"),
                "before_contract_validation": (before_result or {}).get("contract_validation"),
                "after_contract_validation": (after_result or {}).get("contract_validation"),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def _runtime_policy_fired(policy_results: Any) -> bool:
    if not isinstance(policy_results, list):
        return False
    return any(isinstance(item, dict) and item.get("requires_tool") is True for item in policy_results)


def _runtime_effect(result: dict[str, Any]) -> str:
    if result.get("tool_call") is not None:
        return "tool_call"
    if _runtime_policy_fired(result.get("runtime_policy_results", [])):
        return "runtime_policy"
    return "none"


def _numbers(text: str) -> list[str]:
    numbers = []
    for match in re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text):
        numbers.append(match.replace(",", ""))
    return numbers


def _json_answer_matches(expected_text: str, answer_text: str) -> bool | None:
    expected = _parse_json_object(expected_text)
    if expected is None:
        return None
    answer = _parse_json_object(answer_text)
    if answer is None:
        return False
    return answer == expected


def _parse_json_object(text: str) -> Any | None:
    stripped = _strip_json_fence(text.strip())
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _strip_json_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
