from __future__ import annotations

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
    return {"ok": True, "reason": "forced tool call succeeded", "policy_result": forced, "tool_result": tool_result}


def validate_tool_contract(repo_root: Path, tool_path: Path) -> dict[str, Any]:
    tool_name = tool_path.stem
    return validate_tool_tests(repo_root, tool_name)
