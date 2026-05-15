from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agentdistill.config import TaskConfig
from agentdistill.tool_validation import validate_tool_tests
from agentdistill.tools import RuntimePolicyRegistry, ToolRegistry


def validate_runtime_policy_contract(
    repo_root: Path,
    task: TaskConfig,
    policy_path: Path,
) -> dict[str, Any]:
    tools = ToolRegistry(repo_root / "harness" / "tools")
    policies = RuntimePolicyRegistry(policy_path.parent)
    policy_name = policy_path.stem
    payload = {
        "task_instruction": task.instruction,
        "initial_answer": "",
        "tool_call": None,
        "available_tools": tools.names,
        "expected_answer": task.expected_answer,
        "rubric": task.rubric,
    }
    results = [result for result in policies.evaluate(payload) if result.get("policy") == policy_name]
    if not results:
        return {"ok": True, "reason": "policy did not trigger for contract task"}
    forced = results[0]
    if forced.get("requires_tool") is not True:
        return {"ok": True, "reason": "policy does not require a tool for contract task"}
    tool_name = forced.get("tool_name")
    tool_input = forced.get("tool_input")
    if not isinstance(tool_name, str):
        return {"ok": False, "reason": "policy requires tool but tool_name is not a string", "policy_result": forced}
    if not isinstance(tool_input, dict):
        return {"ok": False, "reason": "policy requires tool but tool_input is not an object", "policy_result": forced}
    tool_result = tools.call(tool_name, tool_input)
    if tool_result.get("ok") is not True:
        return {
            "ok": False,
            "reason": "forced tool call did not return ok=true",
            "policy_result": forced,
            "tool_result": tool_result,
        }
    if task.expected_answer and not _tool_result_matches_expected(task.expected_answer, tool_result):
        return {
            "ok": False,
            "reason": "forced tool result does not match expected answer",
            "expected_answer": task.expected_answer,
            "policy_result": forced,
            "tool_result": tool_result,
        }
    return {"ok": True, "reason": "forced tool call succeeded", "policy_result": forced, "tool_result": tool_result}


def validate_tool_contract(repo_root: Path, tool_path: Path) -> dict[str, Any]:
    tool_name = tool_path.stem
    result = validate_tool_tests(repo_root, tool_name)
    if result.get("ok") is not True:
        return result
    tests_path = repo_root / "harness" / "tests" / f"{tool_name}.json"
    return {
        "ok": True,
        "reason": "tool test file exists and all cases passed",
        "tool": tool_name,
        "tests_path": str(tests_path),
        "tool_test_result": result,
    }


def _tool_result_matches_expected(expected_answer: str, tool_result: dict[str, Any]) -> bool:
    expected_numbers = _numbers(expected_answer)
    if not expected_numbers:
        return True
    result_numbers = _numbers(str(tool_result))
    return all(number in result_numbers for number in expected_numbers)


def _numbers(text: str) -> list[str]:
    return [match.replace(",", "") for match in re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)]
