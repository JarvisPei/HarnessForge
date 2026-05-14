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
                "before_failures": ((before_result or {}).get("teacher_diagnosis") or {}).get("failure_categories", []),
                "after_failures": ((after_result or {}).get("teacher_diagnosis") or {}).get("failure_categories", []),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def _numbers(text: str) -> list[str]:
    numbers = []
    for match in re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text):
        numbers.append(match.replace(",", ""))
    return numbers
