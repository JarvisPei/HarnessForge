from __future__ import annotations

import asyncio
from pathlib import Path

from agentdistill.config import TaskConfig, load_benchmark_config
from agentdistill.benchmark import _build_transfer_context, _run_phase
from agentdistill.contracts import (
    validate_runtime_policy_contract,
    validate_runtime_policy_generalization,
    validate_runtime_policy_tests,
    validate_tool_contract,
)
from agentdistill.diagnosis import PatchBundle, parse_diagnosis
from agentdistill.feedback import build_patch_feedback, merge_benchmark_context
from agentdistill.metrics import build_benchmark_metrics
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


def test_contract_validation_rejects_wrong_expected_tool_result(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    tools_dir = harness / "tools"
    policies_dir = harness / "runtime_policies"
    tools_dir.mkdir(parents=True)
    policies_dir.mkdir(parents=True)

    (tools_dir / "wrong_total.py").write_text(
        """
def run(input: dict) -> dict:
    return {"ok": True, "total": 124}
""".strip()
    )
    policy_path = policies_dir / "force_wrong_total.py"
    policy_path.write_text(
        """
def evaluate(input: dict) -> dict:
    return {
        "requires_tool": True,
        "tool_name": "wrong_total",
        "tool_input": {},
        "reason": "bad total"
    }
""".strip()
    )

    result = validate_runtime_policy_contract(
        tmp_path,
        TaskConfig(id="t", instruction="inventory", expected_answer="1,456 labels remain."),
        policy_path,
    )

    assert result["ok"] is False
    assert result["reason"] == "forced tool result does not match expected answer"


def test_runtime_policy_tests_catch_comma_number_parse_errors(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    tools_dir = harness / "tools"
    policies_dir = harness / "runtime_policies"
    tests_dir = harness / "tests"
    tools_dir.mkdir(parents=True)
    policies_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    (tools_dir / "inventory_arithmetic.py").write_text(
        """
def run(input: dict) -> dict:
    total = input["start"] + sum(input["additions"]) - sum(input["subtractions"])
    return {"ok": True, "total": total}
""".strip()
    )
    policy_path = policies_dir / "force_inventory.py"
    policy_path.write_text(
        """
def evaluate(input: dict) -> dict:
    return {
        "requires_tool": True,
        "tool_name": "inventory_arithmetic",
        "tool_input": {"start": 2050, "additions": [720, 288], "subtractions": [138, 1]},
        "reason": "bad comma parse"
    }
""".strip()
    )
    (tests_dir / "force_inventory.json").write_text(
        """
{
  "policy": "force_inventory",
  "cases": [
    {
      "input": {
        "task_instruction": "A store sold 1,107 tags.",
        "available_tools": ["inventory_arithmetic"],
        "expected_answer": "1,813 tags remain."
      },
      "expected": {
        "requires_tool": true,
        "tool_name": "inventory_arithmetic",
        "tool_input": {"subtractions": [138, 1107]}
      },
      "expected_tool_result": {"ok": true, "total": 1813}
    }
  ]
}
""".strip()
    )

    result = validate_runtime_policy_tests(tmp_path, policy_path)
    assert result["ok"] is False
    assert result["reason"] == "one or more policy tests failed"
    assert result["failures"][0]["reason"] == "value mismatch for key: tool_input"


def test_runtime_policy_generalization_rejects_surface_noun_overfit(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    tools_dir = harness / "tools"
    policies_dir = harness / "runtime_policies"
    tests_dir = harness / "tests"
    tools_dir.mkdir(parents=True)
    policies_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    (tools_dir / "inventory_arithmetic.py").write_text(
        """
def run(input: dict) -> dict:
    return {"ok": True, "result": 1813, "unit": "tags", "answer": "1,813 tags remain."}
""".strip()
    )
    policy_path = policies_dir / "force_inventory.py"
    policy_path.write_text(
        """
def evaluate(input: dict) -> dict:
    task = input.get("task_instruction", "").lower()
    if "tags" in task and "remain" in task:
        return {
            "requires_tool": True,
            "tool_name": "inventory_arithmetic",
            "tool_input": {"text": input.get("task_instruction", "")},
            "reason": "narrow trigger"
        }
    return {"requires_tool": False}
""".strip()
    )
    (tests_dir / "force_inventory.json").write_text(
        """
{
  "policy": "force_inventory",
  "cases": [
    {
      "input": {
        "task_instruction": "A store started with 2,050 tags and sold 237 tags. How many tags remain?",
        "available_tools": ["inventory_arithmetic"],
        "expected_answer": "1,813 tags remain."
      },
      "expected": {
        "requires_tool": true,
        "tool_name": "inventory_arithmetic",
        "tool_input": {"text": "A store started with 2,050 tags and sold 237 tags. How many tags remain?"}
      },
      "expected_tool_result": {"ok": true, "result": 1813}
    }
  ]
}
""".strip()
    )

    result = validate_runtime_policy_generalization(tmp_path, policy_path)

    assert result["ok"] is False
    assert result["reason"] == "one or more policy generalization audits failed"
    assert result["failures"][0]["reason"] == "policy trigger is not invariant to schema-preserving wording changes"
    assert result["failures"][0]["mutation"] == "surface_entity_rename"
    assert "daxels" in result["failures"][0]["mutated_instruction"]


def test_runtime_policy_generalization_rejects_operation_verb_overfit(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    tools_dir = harness / "tools"
    policies_dir = harness / "runtime_policies"
    tests_dir = harness / "tests"
    tools_dir.mkdir(parents=True)
    policies_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    (tools_dir / "inventory_arithmetic.py").write_text(
        """
def run(input: dict) -> dict:
    return {"ok": True, "result": 1813, "unit": "tags", "answer": "1,813 tags remain."}
""".strip()
    )
    policy_path = policies_dir / "force_inventory.py"
    policy_path.write_text(
        """
def evaluate(input: dict) -> dict:
    task = input.get("task_instruction", "").lower()
    if "sold" in task and "remain" in task:
        return {
            "requires_tool": True,
            "tool_name": "inventory_arithmetic",
            "tool_input": {"text": input.get("task_instruction", "")},
            "reason": "narrow verb trigger"
        }
    return {"requires_tool": False}
""".strip()
    )
    (tests_dir / "force_inventory.json").write_text(
        """
{
  "policy": "force_inventory",
  "cases": [
    {
      "input": {
        "task_instruction": "A store started with 2,050 tags and sold 237 tags. How many tags remain?",
        "available_tools": ["inventory_arithmetic"],
        "expected_answer": "1,813 tags remain."
      },
      "expected": {
        "requires_tool": true,
        "tool_name": "inventory_arithmetic",
        "tool_input": {"text": "A store started with 2,050 tags and sold 237 tags. How many tags remain?"}
      },
      "expected_tool_result": {"ok": true, "result": 1813}
    }
  ]
}
""".strip()
    )

    result = validate_runtime_policy_generalization(tmp_path, policy_path)

    assert result["ok"] is False
    assert any(
        failure["mutation"] == "direct_subtract_verb"
        and "policy trigger is not invariant" in failure["reason"]
        and "gave away 237" in failure["mutated_instruction"]
        for failure in result["failures"]
    )


def test_runtime_policy_generalization_rejects_removal_phrase_family_gaps(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    tools_dir = harness / "tools"
    policies_dir = harness / "runtime_policies"
    tests_dir = harness / "tests"
    tools_dir.mkdir(parents=True)
    policies_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    (tools_dir / "inventory_arithmetic.py").write_text(
        """
def run(input: dict) -> dict:
    text = input.get("text", "").lower()
    if "sold 237" not in text:
        return {"ok": True, "result": 2050, "unit": "tags", "answer": "2,050 tags remain."}
    return {"ok": True, "result": 1813, "unit": "tags", "answer": "1,813 tags remain."}
""".strip()
    )
    policy_path = policies_dir / "force_inventory.py"
    policy_path.write_text(
        """
def evaluate(input: dict) -> dict:
    task = input.get("task_instruction", "").lower()
    if "started with" in task and "remain" in task:
        return {
            "requires_tool": True,
            "tool_name": "inventory_arithmetic",
            "tool_input": {"text": input.get("task_instruction", "")},
            "reason": "schema trigger with narrow parser"
        }
    return {"requires_tool": False}
""".strip()
    )
    (tests_dir / "force_inventory.json").write_text(
        """
{
  "policy": "force_inventory",
  "cases": [
    {
      "input": {
        "task_instruction": "A store started with 2,050 tags and sold 237 tags. How many tags remain?",
        "available_tools": ["inventory_arithmetic"],
        "expected_answer": "1,813 tags remain."
      },
      "expected": {
        "requires_tool": true,
        "tool_name": "inventory_arithmetic",
        "tool_input": {"text": "A store started with 2,050 tags and sold 237 tags. How many tags remain?"}
      },
      "expected_tool_result": {"ok": true, "result": 1813}
    }
  ]
}
""".strip()
    )

    result = validate_runtime_policy_generalization(tmp_path, policy_path)

    assert result["ok"] is False
    assert any(
        failure["mutation"] == "removal_phrase_family"
        and failure["reason"] == "forced tool result failed after schema-preserving wording changes"
        and (
            "handed out 237" in failure["mutated_instruction"]
            or "redeemed 237" in failure["mutated_instruction"]
            or "voided 237" in failure["mutated_instruction"]
        )
        for failure in result["failures"]
    )


def test_runtime_policy_generalization_accepts_schema_trigger(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    tools_dir = harness / "tools"
    policies_dir = harness / "runtime_policies"
    tests_dir = harness / "tests"
    tools_dir.mkdir(parents=True)
    policies_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    (tools_dir / "inventory_arithmetic.py").write_text(
        """
def run(input: dict) -> dict:
    return {"ok": True, "result": 1813, "unit": "items", "answer": "1,813 items remain."}
""".strip()
    )
    policy_path = policies_dir / "force_inventory.py"
    policy_path.write_text(
        """
def evaluate(input: dict) -> dict:
    task = input.get("task_instruction", "").lower()
    if "started with" in task and "remain" in task:
        return {
            "requires_tool": True,
            "tool_name": "inventory_arithmetic",
            "tool_input": {"text": input.get("task_instruction", "")},
            "reason": "schema trigger"
        }
    return {"requires_tool": False}
""".strip()
    )
    (tests_dir / "force_inventory.json").write_text(
        """
{
  "policy": "force_inventory",
  "cases": [
    {
      "input": {
        "task_instruction": "A store started with 2,050 tags and sold 237 tags. How many tags remain?",
        "available_tools": ["inventory_arithmetic"],
        "expected_answer": "1,813 tags remain."
      },
      "expected": {
        "requires_tool": true,
        "tool_name": "inventory_arithmetic",
        "tool_input": {"text": "A store started with 2,050 tags and sold 237 tags. How many tags remain?"}
      },
      "expected_tool_result": {"ok": true, "result": 1813}
    }
  ]
}
""".strip()
    )

    result = validate_runtime_policy_generalization(tmp_path, policy_path)

    assert result["ok"] is True
    assert result["reason"] == "policy generalization audit passed"
    assert result["num_cases"] >= 2


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
  "harness_manifest": {
    "bundle_id": "adder_bundle",
    "intent": "Add a deterministic addition tool.",
    "allowed_paths": ["harness/tests/adder.json", "harness/tools/adder.py"],
    "artifacts": [
      {"path": "harness/tests/adder.json", "type": "test", "purpose": "tool tests"},
      {"path": "harness/tools/adder.py", "type": "tool", "purpose": "addition helper"}
    ],
    "contracts": ["tool tests pass"]
  },
  "confidence": 0.8
}
""".strip()
    )

    assert len(diagnosis.patch_bundles) == 2
    assert diagnosis.patch_bundle is not None
    assert diagnosis.patch_bundle.target_path == "harness/tests/adder.json"
    assert diagnosis.harness_manifest is not None
    assert diagnosis.harness_manifest.bundle_id == "adder_bundle"


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
            PatchBundle(
                target_path="harness/tests/force_adder.json",
                action="create_or_replace",
                content="""
{
  "policy": "force_adder",
  "cases": [
    {
      "input": {
        "task_instruction": "add 2 and 3",
        "available_tools": ["adder"],
        "expected_answer": "5"
      },
      "expected": {
        "requires_tool": true,
        "tool_name": "adder",
        "tool_input": {"a": 2, "b": 3}
      },
      "expected_tool_result": {"ok": true, "total": 5}
    }
  ]
}
""".strip(),
                rationale="Covers the runtime policy contract.",
            ),
        ],
        task,
        _manifest_for(
            [
                "harness/tools/adder.py",
                "harness/tests/adder.json",
                "harness/runtime_policies/force_adder.py",
                "harness/tests/force_adder.json",
            ]
        ),
    )

    assert result["patch_status"] == "accepted"
    assert len(result["applied_patch_paths"]) == 4
    assert all(contract["ok"] is True for contract in result["contract_validation"])
    assert (tmp_path / "harness" / "tools" / "adder.py").exists()
    assert (tmp_path / "outputs" / "harness_workspaces" / "adder_bundle" / "harness" / "tools" / "adder.py").exists()


def test_code_patch_bundles_require_manifest(tmp_path: Path) -> None:
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
            )
        ],
        task,
    )

    assert result["patch_status"] == "rejected"
    assert result["rejection_reason"] == "harness manifest validation failed"
    assert "code harness bundles must include harness_manifest" in result["contract_validation"][0]["reason"]
    assert result["rejected_patch_paths"] == [str(tmp_path / "harness" / "tools" / "adder.py")]
    assert not (tmp_path / "harness" / "tools" / "adder.py").exists()


def test_runtime_policy_patch_requires_matching_policy_tests(tmp_path: Path) -> None:
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
                content='{"tool":"adder","cases":[{"input":{"a":2,"b":3},"expected":{"ok":true,"total":5}}]}',
                rationale="Tool tests.",
            ),
            PatchBundle(
                target_path="harness/runtime_policies/force_adder.py",
                action="create_or_replace",
                content="""
def evaluate(input: dict) -> dict:
    return {"requires_tool": True, "tool_name": "adder", "tool_input": {"a": 2, "b": 3}}
""".strip(),
                rationale="Force tool use.",
            ),
        ],
        task,
        _manifest_for(
            [
                "harness/tools/adder.py",
                "harness/tests/adder.json",
                "harness/runtime_policies/force_adder.py",
            ]
        ),
    )

    assert result["patch_status"] == "rejected"
    assert any("no matching JSON test file found" in item.get("reason", "") for item in result["contract_validation"])
    assert not (tmp_path / "harness" / "runtime_policies" / "force_adder.py").exists()


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
        _manifest_for(["harness/tools/adder.py", "harness/tests/adder.json"]),
    )

    assert result["patch_status"] == "rejected"
    assert "one or more patch contracts failed" in result["rejection_reason"]
    assert result["rejected_patch_paths"] == [
        str(tmp_path / "harness" / "tools" / "adder.py"),
        str(tmp_path / "harness" / "tests" / "adder.json"),
    ]
    assert not (tmp_path / "harness" / "tools" / "adder.py").exists()
    assert not (tmp_path / "harness" / "tests" / "adder.json").exists()


def _make_harness_dirs(root: Path) -> None:
    for name in ["guidelines", "skills", "validators", "tools", "runtime_policies", "tests"]:
        (root / "harness" / name).mkdir(parents=True, exist_ok=True)


def _manifest_for(paths: list[str]):
    from agentdistill.manifest import HarnessManifest

    artifact_types = {
        "guidelines": "guideline",
        "skills": "skill",
        "validators": "validator",
        "tools": "tool",
        "tests": "test",
        "runtime_policies": "runtime_policy",
    }
    return HarnessManifest.model_validate(
        {
            "bundle_id": "adder_bundle",
            "intent": "Add deterministic addition as a reusable harness tool.",
            "allowed_paths": paths,
            "artifacts": [
                {"path": path, "type": artifact_types[Path(path).parts[1]], "purpose": "test artifact"}
                for path in paths
            ],
            "contracts": ["tool tests pass", "runtime policy forced tool result matches expected answer"],
        }
    )


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


def test_patch_feedback_summarizes_rejected_contract_failures() -> None:
    feedback = build_patch_feedback(
        {
            "train_inventory": {
                "patch_status": "rejected",
                "rejection_reason": "one or more patch contracts failed",
                "rejected_patch_paths": ["/repo/harness/runtime_policies/force_inventory.py"],
                "harness_manifest": {"bundle_id": "inventory_policy"},
                "contract_validation": [
                    {"ok": True, "reason": "manifest matches patch bundle"},
                    {
                        "ok": False,
                        "path": "/repo/harness/runtime_policies/force_inventory.py",
                        "reason": "one or more policy tests failed",
                        "policy": "force_inventory",
                        "failures": [
                            {
                                "case_index": 0,
                                "reason": "value mismatch for key: tool_input",
                                "expected": {"tool_input": {"subtractions": [138, 1107]}},
                                "actual": {"tool_input": {"subtractions": [138, 1]}},
                            }
                        ],
                    },
                ],
            },
            "train_ok": {"patch_status": "accepted", "contract_validation": [{"ok": True}]},
        },
        iteration=1,
    )

    assert feedback["has_rejections"] is True
    rejected = feedback["rejected_bundles"][0]
    assert rejected["bundle_id"] == "inventory_policy"
    assert rejected["failed_contracts"][0]["reason"] == "one or more policy tests failed"
    assert rejected["failed_contracts"][0]["failures"][0]["actual"]["tool_input"]["subtractions"] == [138, 1]


def test_teacher_prompt_uses_meta_skills_not_domain_scaffolds() -> None:
    prompt = Path("prompts/teacher_diagnosis.md").read_text()

    assert "Meta-Skill: Parser Design" in prompt
    assert "Meta-Skill: Tool Interface Design" in prompt
    assert "Meta-Skill: Runtime Policy Test Design" in prompt
    assert "Meta-Skill: Runtime Policy Trigger Design" in prompt
    assert "Meta-Skill: Contract Repair" in prompt
    assert "Meta-Skill: Generalization Discipline" in prompt
    assert "ADD_WORDS" not in prompt
    assert "SUBTRACT_WORDS" not in prompt
    assert "_parse_inventory" not in prompt
    assert "For inventory arithmetic runtime policies" not in prompt
    assert "Inventory policy tests must include" not in prompt
    assert "For unit conversion tasks" not in prompt
    assert "Unit conversion policy tests must include" not in prompt


def test_merge_benchmark_context_adds_patch_feedback_only_for_rejections() -> None:
    transfer_context = {"heldout_probe": [{"task_id": "dev"}]}
    empty_feedback = {"iteration": 1, "has_rejections": False, "rejected_bundles": []}
    rejected_feedback = {"iteration": 1, "has_rejections": True, "rejected_bundles": [{"task_id": "train"}]}

    assert "patch_feedback" not in merge_benchmark_context(transfer_context, empty_feedback)
    merged = merge_benchmark_context(transfer_context, rejected_feedback)
    assert merged["heldout_probe"] == transfer_context["heldout_probe"]
    assert merged["patch_feedback"] == rejected_feedback


def test_run_phase_records_context_patch_feedback(tmp_path: Path) -> None:
    class WeakClient:
        async def complete(self, messages, temperature=0.2):
            return "0"

    class TeacherClient:
        async def complete(self, messages, temperature=0.2):
            return '{"failure_categories":[],"patch_type":"prompt_guideline","patch_bundles":[]}'

    class HarnessConfig:
        system_prompt_path = tmp_path / "weak_system.md"
        skills_dir = tmp_path / "harness" / "skills"
        guidelines_dir = tmp_path / "harness" / "guidelines"
        validators_dir = tmp_path / "harness" / "validators"
        tools_dir = tmp_path / "harness" / "tools"
        runtime_policies_dir = tmp_path / "harness" / "runtime_policies"

    class Config:
        name = "test"
        harness = HarnessConfig()

    for directory in [
        HarnessConfig.skills_dir,
        HarnessConfig.guidelines_dir,
        HarnessConfig.validators_dir,
        HarnessConfig.tools_dir,
        HarnessConfig.runtime_policies_dir,
    ]:
        directory.mkdir(parents=True)
    HarnessConfig.system_prompt_path.write_text("system")

    context = {"heldout_probe": [], "patch_feedback": {"has_rejections": True, "rejected_bundles": [{"task_id": "t"}]}}
    results = asyncio.run(
        _run_phase(
            Config(),
            "train",
            [TaskConfig(id="t", instruction="answer", expected_answer="1")],
            WeakClient(),
            TeacherClient(),
            "teacher",
            tmp_path / "outputs",
            False,
            tmp_path,
            benchmark_context=context,
        )
    )

    assert results["t"]["context_patch_feedback"] == context["patch_feedback"]


def test_benchmark_config_splits_dev_and_blind_tasks(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "bench.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
name: split_benchmark
output_dir: outputs/split
weak: {role: weak}
teacher: {role: teacher}
harness:
  system_prompt_path: prompts/weak_system.md
train_tasks:
  - id: train
    instruction: train
dev_probe_tasks:
  - id: dev
    instruction: dev
blind_test_tasks:
  - id: blind
    instruction: blind
""".strip()
    )

    cfg = load_benchmark_config(config_path)
    assert [task.id for task in cfg.dev_probe_tasks] == ["dev"]
    assert [task.id for task in cfg.blind_test_tasks] == ["blind"]


def test_benchmark_config_keeps_heldout_fallback(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "bench.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
name: legacy_benchmark
output_dir: outputs/legacy
weak: {role: weak}
teacher: {role: teacher}
harness:
  system_prompt_path: prompts/weak_system.md
train_tasks:
  - id: train
    instruction: train
heldout_tasks:
  - id: heldout
    instruction: heldout
""".strip()
    )

    cfg = load_benchmark_config(config_path)
    assert [task.id for task in cfg.dev_probe_tasks] == ["heldout"]
    assert [task.id for task in cfg.blind_test_tasks] == ["heldout"]


def test_inventory_benchmark_allows_three_repair_iterations() -> None:
    cfg = load_benchmark_config("configs/benchmark_inventory.yaml")
    assert cfg.evolve_iterations == 3
    assert [task.id for task in cfg.dev_probe_tasks] == ["heldout_inventory_tags", "heldout_inventory_stickers"]
    assert [task.id for task in cfg.blind_test_tasks] == ["blind_inventory_badges", "blind_inventory_vouchers"]


def test_unit_conversion_benchmark_config() -> None:
    cfg = load_benchmark_config("configs/benchmark_unit_conversion.yaml")
    assert cfg.name == "benchmark_unit_conversion"
    assert cfg.evolve_iterations == 3
    assert [task.id for task in cfg.train_tasks] == ["train_solution_liters"]
    assert [task.id for task in cfg.dev_probe_tasks] == ["dev_package_grams", "dev_cable_meters"]
    assert [task.id for task in cfg.blind_test_tasks] == ["blind_syrup_milliliters", "blind_rope_centimeters"]


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


def test_build_benchmark_metrics_counts_patch_quality_and_transfer() -> None:
    metrics = build_benchmark_metrics(
        train_summary=[
            {
                "patch_status": "accepted",
                "applied_patch_paths": [
                    "/repo/harness/tools/inventory.py",
                    "/repo/harness/tests/inventory.json",
                    "/repo/harness/runtime_policies/force_inventory.py",
                    "/repo/harness/guidelines/inventory.md",
                ],
                "contract_validation": [{"ok": True}],
                "harness_manifest": {
                    "bundle_id": "inventory_bundle",
                    "artifacts": [
                        {"path": "/repo/harness/tools/inventory.py", "type": "tool"},
                        {"path": "/repo/harness/tests/inventory.json", "type": "test"},
                    ],
                },
            },
            {
                "patch_status": "rejected",
                "applied_patch_paths": [],
                "rejected_patch_paths": ["/repo/harness/runtime_policies/bad.py"],
                "contract_validation": [{"ok": False}],
            },
        ],
        impact_rows=[
            {"before_success": False, "after_success": True, "improved": True, "regressed": False},
            {"before_success": True, "after_success": True, "improved": False, "regressed": False},
        ],
        harness_files_after=[
            "harness/tools/inventory.py",
            "harness/tests/inventory.json",
            "harness/runtime_policies/force_inventory.py",
        ],
    )

    assert metrics["patches"]["accepted"] == 1
    assert metrics["patches"]["rejected"] == 1
    assert metrics["patches"]["accepted_tool_test_policy_bundles"] == 1
    assert metrics["patches"]["accepted_code_manifest_bundles"] == 1
    assert metrics["patches"]["contract_failures"] == 1
    assert metrics["transfer"]["improved"] == 1
    assert metrics["dev_transfer"]["improved"] == 1
    assert metrics["blind_transfer"]["improved"] == 1
    assert metrics["harness_after"]["type_counts"]["tool"] == 1


def test_build_benchmark_metrics_separates_blind_transfer() -> None:
    metrics = build_benchmark_metrics(
        train_summary=[],
        impact_rows=[{"before_success": False, "after_success": True, "improved": True, "regressed": False}],
        harness_files_after=[],
        blind_impact_rows=[{"before_success": False, "after_success": False, "improved": False, "regressed": False}],
    )

    assert metrics["dev_transfer"]["improved"] == 1
    assert metrics["blind_transfer"]["improved"] == 0
