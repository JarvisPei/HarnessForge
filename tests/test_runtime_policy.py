from __future__ import annotations

from pathlib import Path

from agentdistill.config import TaskConfig
from agentdistill.contracts import validate_runtime_policy_contract, validate_tool_contract
from agentdistill.tools import RuntimePolicyRegistry, ToolRegistry


def test_runtime_policy_can_force_tool_call(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    policies_dir = tmp_path / "runtime_policies"
    tools_dir.mkdir()
    policies_dir.mkdir()

    (tools_dir / "adder.py").write_text(
        """
def run(input: dict) -> dict:
    return {"ok": True, "total": input["a"] + input["b"]}
""".strip()
    )
    (policies_dir / "force_adder.py").write_text(
        """
def evaluate(input: dict) -> dict:
    if "adder" in input.get("available_tools", []):
        return {
            "requires_tool": True,
            "tool_name": "adder",
            "tool_input": {"a": 2, "b": 3},
            "reason": "Use deterministic addition."
        }
    return {"requires_tool": False}
""".strip()
    )

    tools = ToolRegistry(tools_dir)
    policies = RuntimePolicyRegistry(policies_dir)
    results = policies.evaluate(
        {
            "task_instruction": "add 2 and 3",
            "initial_answer": "4",
            "tool_call": None,
            "available_tools": tools.names,
        }
    )

    assert results[0]["requires_tool"] is True
    assert results[0]["tool_name"] == "adder"
    assert tools.call("adder", results[0]["tool_input"])["total"] == 5


def test_policy_rejects_banned_import(tmp_path: Path) -> None:
    policies_dir = tmp_path / "runtime_policies"
    policies_dir.mkdir()
    (policies_dir / "bad.py").write_text(
        """
import os

def evaluate(input: dict) -> dict:
    return {"requires_tool": False}
""".strip()
    )

    try:
        RuntimePolicyRegistry(policies_dir)
    except RuntimeError as exc:
        assert "banned module" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for banned import")


def test_contract_validation_rejects_bad_tool_input(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    tools_dir = harness / "tools"
    policies_dir = harness / "runtime_policies"
    tools_dir.mkdir(parents=True)
    policies_dir.mkdir(parents=True)

    (tools_dir / "needs_x.py").write_text(
        """
def run(input: dict) -> dict:
    if "x" not in input:
        return {"ok": False, "error": "missing x"}
    return {"ok": True, "x": input["x"]}
""".strip()
    )
    policy_path = policies_dir / "bad_policy.py"
    policy_path.write_text(
        """
def evaluate(input: dict) -> dict:
    return {
        "requires_tool": True,
        "tool_name": "needs_x",
        "tool_input": {"task_instruction": input.get("task_instruction", "")},
        "reason": "bad schema"
    }
""".strip()
    )

    result = validate_runtime_policy_contract(
        tmp_path,
        TaskConfig(id="t", instruction="use tool", expected_answer="ok"),
        policy_path,
    )

    assert result["ok"] is False
    assert "tool_result" in result


def test_tool_contract_validation_runs_json_tests(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    tools_dir = harness / "tools"
    tests_dir = harness / "tests"
    tools_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    tool_path = tools_dir / "adder.py"
    tool_path.write_text(
        """
def run(input: dict) -> dict:
    return {"ok": True, "total": input["a"] + input["b"]}
""".strip()
    )
    (tests_dir / "adder.json").write_text(
        """
{
  "tool": "adder",
  "cases": [
    {
      "input": {"a": 2, "b": 3},
      "expected": {"ok": true, "total": 5}
    }
  ]
}
""".strip()
    )

    result = validate_tool_contract(tmp_path, tool_path)
    assert result["ok"] is True
