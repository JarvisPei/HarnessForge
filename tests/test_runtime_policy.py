from __future__ import annotations

from pathlib import Path

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
