from __future__ import annotations

import json
import re
from collections import Counter
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


def build_tau_run_report(run_dir: Path, output_path: Path | None = None) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [_tau_trace_report_row(run_dir, trace_summary) for trace_summary in summary.get("traces", [])]
    report = {
        "run_dir": str(run_dir),
        "domain": summary.get("domain"),
        "split": summary.get("split"),
        "task_ids": summary.get("task_ids", []),
        "aggregate": _tau_aggregate(rows),
        "tasks": rows,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _tau_trace_report_row(run_dir: Path, trace_summary: dict[str, Any]) -> dict[str, Any]:
    trace = _load_tau_trace(run_dir, trace_summary)
    error_info = _tau_error_info(trace, trace_summary)
    reward_info = trace_summary.get("reward_info")
    if not isinstance(reward_info, dict):
        reward_info = trace.get("reward_info") if isinstance(trace.get("reward_info"), dict) else {}
    reward = reward_info.get("reward")
    action_checks = reward_info.get("action_checks")
    checks = [check for check in action_checks if isinstance(check, dict)] if isinstance(action_checks, list) else []
    unmatched_checks = [check for check in checks if check.get("action_match") is not True]
    write_checks = [check for check in checks if check.get("tool_type") == "write"]
    unmatched_write_checks = [check for check in write_checks if check.get("action_match") is not True]
    actual_tool_calls = _tau_actual_tool_calls(trace)
    actual_write_tool_calls = [call for call in actual_tool_calls if _tau_tool_name_is_write(call.get("name"))]
    official_pass = reward == 1.0
    strict_action_pass = None if not checks else not unmatched_checks
    return {
        "task_id": trace_summary.get("task_id") or trace.get("task_id"),
        "termination_reason": trace_summary.get("termination_reason") or trace.get("termination_reason"),
        "official_reward": reward,
        "official_pass": official_pass,
        "strict_action_pass": strict_action_pass,
        "official_pass_but_action_mismatch": official_pass and strict_action_pass is False,
        "action_checks_total": len(checks),
        "action_checks_matched": len(checks) - len(unmatched_checks),
        "action_checks_unmatched": len(unmatched_checks),
        "unmatched_actions": [_tau_action_brief(check) for check in unmatched_checks],
        "expected_write_actions": len(write_checks),
        "matched_write_actions": len(write_checks) - len(unmatched_write_checks),
        "unmatched_write_actions": len(unmatched_write_checks),
        "actual_tool_calls_total": len(actual_tool_calls),
        "actual_write_tool_calls": len(actual_write_tool_calls),
        "actual_write_tool_names": [call.get("name") for call in actual_write_tool_calls],
        "actual_cancel_reservation_ids": [
            call.get("arguments", {}).get("reservation_id")
            for call in actual_tool_calls
            if call.get("name") == "cancel_reservation" and isinstance(call.get("arguments"), dict)
        ],
        "error_type": error_info.get("type"),
        "error_category": _tau_error_category(error_info),
        "error_message": _truncate_text(error_info.get("message"), limit=300),
        "path": trace_summary.get("path"),
    }


def _tau_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [row["official_reward"] for row in rows if isinstance(row.get("official_reward"), (int, float))]
    strict_rows = [row for row in rows if row.get("strict_action_pass") is not None]
    official_passed = sum(1 for row in rows if row.get("official_pass") is True)
    adapter_error_rows = [row for row in rows if row.get("termination_reason") == "adapter_error"]
    error_categories = Counter(
        row.get("error_category") for row in adapter_error_rows if isinstance(row.get("error_category"), str)
    )
    return {
        "num_tasks": len(rows),
        "official_passed": official_passed,
        "official_failed": len(rows) - official_passed,
        "official_reward_sum": round(float(sum(rewards)), 4),
        "official_reward_mean": round(float(sum(rewards) / len(rewards)), 4) if rewards else None,
        "timeouts": sum(1 for row in rows if row.get("termination_reason") == "timeout"),
        "adapter_errors": len(adapter_error_rows),
        "adapter_error_categories": dict(sorted(error_categories.items())),
        "strict_action_evaluated": len(strict_rows),
        "strict_action_passed": sum(1 for row in strict_rows if row.get("strict_action_pass") is True),
        "strict_action_unmatched_tasks": sum(1 for row in strict_rows if row.get("strict_action_pass") is False),
        "official_pass_but_action_mismatch": sum(
            1 for row in rows if row.get("official_pass_but_action_mismatch") is True
        ),
        "expected_write_actions": sum(int(row.get("expected_write_actions", 0)) for row in rows),
        "matched_write_actions": sum(int(row.get("matched_write_actions", 0)) for row in rows),
        "unmatched_write_actions": sum(int(row.get("unmatched_write_actions", 0)) for row in rows),
        "actual_write_tool_calls": sum(int(row.get("actual_write_tool_calls", 0)) for row in rows),
    }


def _load_tau_trace(run_dir: Path, trace_summary: dict[str, Any]) -> dict[str, Any]:
    path = trace_summary.get("path")
    if not isinstance(path, str):
        return {}
    trace_path = run_dir / path
    if not trace_path.exists():
        return {}
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return trace if isinstance(trace, dict) else {}


def _tau_error_info(trace: dict[str, Any], trace_summary: dict[str, Any]) -> dict[str, Any]:
    error = trace.get("error")
    if not isinstance(error, dict):
        error = trace_summary.get("error")
    return error if isinstance(error, dict) else {}


def _tau_error_category(error: dict[str, Any]) -> str | None:
    error_type = str(error.get("type") or "")
    message = str(error.get("message") or "")
    lowered = f"{error_type} {message}".lower()
    if not lowered.strip():
        return None
    if "httpstatuserror" in lowered or "transporterror" in lowered or "timeout" in lowered:
        return "model_transport"
    if "empty_response" in lowered or "no response received from upstream" in lowered:
        return "model_transport"
    if "usermessage must have either content or tool_calls" in lowered:
        return "user_simulator_empty_message"
    return "adapter_error"


def _truncate_text(value: Any, *, limit: int) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _tau_actual_tool_calls(trace: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    messages = trace.get("messages")
    if not isinstance(messages, list):
        return calls
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if isinstance(call, dict):
                calls.append(call)
    return calls


def _tau_tool_name_is_write(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    return not name.startswith(("get_", "list_", "search_", "find_"))


def _tau_action_brief(check: dict[str, Any]) -> dict[str, Any]:
    action = check.get("action")
    action = action if isinstance(action, dict) else {}
    return {
        "action_id": action.get("action_id"),
        "name": action.get("name"),
        "arguments": action.get("arguments"),
        "tool_type": check.get("tool_type"),
    }


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
