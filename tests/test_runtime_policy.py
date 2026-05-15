from __future__ import annotations

import asyncio
from pathlib import Path

from agentdistill.config import TaskConfig
from agentdistill.benchmark import _build_transfer_context
from agentdistill.contracts import validate_runtime_policy_contract, validate_tool_contract
from agentdistill.diagnosis import PatchBundle, parse_diagnosis
from agentdistill.patches import apply_patch_bundles_atomically
from agentdistill.models import load_model_settings
from agentdistill.run import run_task
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


def test_tool_contract_requires_matching_json_tests(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    tools_dir = harness / "tools"
    tools_dir.mkdir(parents=True)
    tool_path = tools_dir / "adder.py"
    tool_path.write_text(
        """
def run(input: dict) -> dict:
    return {"ok": True, "total": input["a"] + input["b"]}
""".strip()
    )

    result = validate_tool_contract(tmp_path, tool_path)
    assert result["ok"] is False
    assert "no matching tool test" in result["reason"]


def test_parse_diagnosis_accepts_patch_bundles() -> None:
    diagnosis = parse_diagnosis(
        """
{
  "diagnosis": "Need a grouped harness update.",
  "failure_categories": ["tool", "runtime_policy"],
  "harness_patch": "Add a tool, tests, and policy.",
  "patch_type": "tool",
  "regression_test": "The grouped patch should parse.",
  "patch_bundles": [
    {
      "target_path": "harness/tests/adder.json",
      "action": "create_or_replace",
      "content": "{\\"tool\\": \\"adder\\", \\"cases\\": []}",
      "rationale": "test"
    },
    {
      "target_path": "harness/tools/adder.py",
      "action": "create_or_replace",
      "content": "def run(input: dict) -> dict:\\n    return {\\"ok\\": True}",
      "rationale": "tool"
    }
  ],
  "confidence": 0.8
}
""".strip()
    )

    assert len(diagnosis.patch_bundles) == 2
    assert diagnosis.patch_bundle is not None
    assert diagnosis.patch_bundle.target_path == "harness/tests/adder.json"


def test_parse_diagnosis_allows_missing_bundle_rationale() -> None:
    diagnosis = parse_diagnosis(
        """
{
  "diagnosis": "Missing rationale should not break parsing.",
  "failure_categories": ["tool"],
  "harness_patch": "Add a tool.",
  "patch_type": "tool",
  "regression_test": "Parser should accept omitted rationale.",
  "patch_bundles": [
    {
      "target_path": "harness/tools/adder.py",
      "action": "create_or_replace",
      "content": "def run(input: dict) -> dict:\\n    return {\\"ok\\": True}"
    }
  ],
  "confidence": 0.5
}
""".strip()
    )

    assert diagnosis.patch_bundles[0].rationale == ""


def test_parse_diagnosis_allows_lean_patch_payloads() -> None:
    diagnosis = parse_diagnosis(
        """
{
  "failure_categories": ["prompt_guideline"],
  "patch_type": "prompt_guideline",
  "patch_bundles": []
}
""".strip()
    )

    assert diagnosis.diagnosis
    assert diagnosis.harness_patch == ""


def test_parse_diagnosis_returns_unparsed_for_malformed_json() -> None:
    diagnosis = parse_diagnosis('{"diagnosis":"x" "failure_categories":[]}')
    assert diagnosis.parse_status == "unparsed"
    assert diagnosis.patch_type == "unparsed"


def test_parse_diagnosis_handles_fenced_and_nested_json() -> None:
    diagnosis = parse_diagnosis(
        """
Here is the diagnosis:
```json
{
  "diagnosis": "Nested braces in strings should not break parsing.",
  "failure_categories": ["tool"],
  "harness_patch": "Use text like {not json} in the explanation.",
  "patch_type": "tool",
  "regression_test": "Parser should handle long payloads.",
  "patch_bundles": [
    {
      "target_path": "harness/tools/adder.py",
      "action": "create_or_replace",
      "content": "def run(input: dict) -> dict:\\n    return {\\"ok\\": True, \\"text\\": \\"{brace}\\"}",
      "rationale": "keep braces"
    }
  ],
  "confidence": 0.9
}
```
Trailing text.
""".strip()
    )

    assert diagnosis.diagnosis.startswith("Nested braces")
    assert len(diagnosis.patch_bundles) == 1


def test_atomic_patch_bundles_accept_tool_tests_and_policy(tmp_path: Path) -> None:
    _make_harness_dirs(tmp_path)
    task = TaskConfig(id="t", instruction="add 2 and 3", expected_answer="5")

    result = apply_patch_bundles_atomically(
        tmp_path,
        [
            PatchBundle(
                target_path="harness/tools/adder.py",
                action="create_or_replace",
                content="""
def run(input: dict) -> dict:
    return {"ok": True, "total": input["a"] + input["b"]}
""".strip(),
                rationale="Deterministic addition.",
            ),
            PatchBundle(
                target_path="harness/tests/adder.json",
                action="create_or_replace",
                content="""
{
  "tool": "adder",
  "cases": [
    {
      "input": {"a": 2, "b": 3},
      "expected": {"ok": true, "total": 5}
    }
  ]
}
""".strip(),
                rationale="Covers the tool contract.",
            ),
            PatchBundle(
                target_path="harness/runtime_policies/force_adder.py",
                action="create_or_replace",
                content="""
def evaluate(input: dict) -> dict:
    return {
        "requires_tool": True,
        "tool_name": "adder",
        "tool_input": {"a": 2, "b": 3},
        "reason": "Use exact arithmetic."
    }
""".strip(),
                rationale="Forces exact arithmetic.",
            ),
        ],
        task,
    )

    assert result["patch_status"] == "accepted"
    assert len(result["applied_patch_paths"]) == 3
    assert all(contract["ok"] is True for contract in result["contract_validation"])
    assert (tmp_path / "harness" / "tools" / "adder.py").exists()


def test_atomic_patch_bundles_roll_back_group_on_failed_tool_tests(tmp_path: Path) -> None:
    _make_harness_dirs(tmp_path)
    task = TaskConfig(id="t", instruction="add 2 and 3", expected_answer="5")

    result = apply_patch_bundles_atomically(
        tmp_path,
        [
            PatchBundle(
                target_path="harness/tools/adder.py",
                action="create_or_replace",
                content="""
def run(input: dict) -> dict:
    return {"ok": True, "total": input["a"] + input["b"]}
""".strip(),
                rationale="Deterministic addition.",
            ),
            PatchBundle(
                target_path="harness/tests/adder.json",
                action="create_or_replace",
                content="""
{
  "tool": "adder",
  "cases": [
    {
      "input": {"a": 2, "b": 3},
      "expected": {"ok": true, "total": 6}
    }
  ]
}
""".strip(),
                rationale="Bad test expectation should reject the group.",
            ),
        ],
        task,
    )

    assert result["patch_status"] == "rejected"
    assert "one or more patch contracts failed" in result["rejection_reason"]
    assert not (tmp_path / "harness" / "tools" / "adder.py").exists()
    assert not (tmp_path / "harness" / "tests" / "adder.json").exists()


def _make_harness_dirs(root: Path) -> None:
    for name in ["guidelines", "skills", "validators", "tools", "runtime_policies", "tests"]:
        (root / "harness" / name).mkdir(parents=True, exist_ok=True)


def test_load_model_settings_prefers_role_specific_timeout(monkeypatch) -> None:
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("TEACHER_TIMEOUT_SECONDS", "33")
    monkeypatch.setenv("WEAK_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("TEACHER_PROVIDER", "openai")
    monkeypatch.setenv("TEACHER_BASE_URL", "https://example.com")
    monkeypatch.setenv("TEACHER_API_KEY", "k")
    monkeypatch.setenv("TEACHER_MODEL", "m")
    settings = load_model_settings("teacher")
    assert settings.timeout_seconds == 33.0


def test_build_transfer_context_uses_heldout_probe_results() -> None:
    tasks = [
        TaskConfig(id="heldout_a", instruction="a", expected_answer="1"),
        TaskConfig(id="heldout_b", instruction="b", expected_answer="2"),
    ]
    results = {
        "heldout_a": {
            "weak_answer": "1",
            "tool_call": {"name": "adder", "input": {"a": 1}},
            "runtime_policy_results": [{"requires_tool": False}],
        }
    }

    context = _build_transfer_context(tasks, results)
    assert context["heldout_probe"][0]["success"] is True
    assert context["heldout_probe"][0]["weak_answer"] == "1"
    assert context["heldout_probe"][1]["success"] is False


def test_run_task_can_skip_teacher_diagnosis(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    policies_dir = tmp_path / "runtime_policies"
    tools_dir.mkdir()
    policies_dir.mkdir()

    class WeakClient:
        async def complete(self, messages, temperature=0.2):
            return "1"

    class TeacherClient:
        async def complete(self, messages, temperature=0.2):
            raise AssertionError("teacher should not be called for weak-only probes")

    result = asyncio.run(
        run_task(
            TaskConfig(id="heldout", instruction="answer 1", expected_answer="1"),
            WeakClient(),
            TeacherClient(),
            "system",
            "teacher",
            ToolRegistry(tools_dir),
            RuntimePolicyRegistry(policies_dir),
            request_teacher_diagnosis=False,
        )
    )

    assert result["weak_answer"] == "1"
    assert "teacher_diagnosis_raw" not in result
