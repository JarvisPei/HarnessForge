from __future__ import annotations

import asyncio
import json
from pathlib import Path

import agentdistill.benchmark as benchmark_module
from agentdistill.config import TaskConfig, load_benchmark_config
from agentdistill.benchmark import (
    _benchmark_context_for_iteration,
    _build_focused_repair_task,
    _build_transfer_context,
    _critic_enabled,
    _initial_transfer_context,
    _infer_repair_scope,
    _phase_kind,
    _reject_out_of_scope_repair,
    _run_focused_repair_task,
    _run_inner_repair_attempts,
    _run_phase,
    _should_request_critic_cases,
    _tasks_for_evolve_iteration,
)
from agentdistill.contracts import (
    validate_runtime_policy_contract,
    validate_runtime_policy_generalization,
    validate_runtime_policy_tests,
    validate_tool_contract,
)
from agentdistill.critic import parse_critic_audit, validate_critic_policy_cases
from agentdistill.diagnosis import PatchBundle, parse_diagnosis
from agentdistill.feedback import build_patch_feedback, build_transfer_feedback, merge_benchmark_context
from agentdistill.metrics import build_benchmark_metrics
from agentdistill.report import build_impact_report
from agentdistill.repair_probe import run_repair_probe
from agentdistill.repair_family import _weak_system, build_probe_filter_cases, build_repair_family_cases, run_probe_filter, run_repair_family
from agentdistill.patches import apply_patch_bundles_atomically
from agentdistill.patches import patch_group_is_executable
from agentdistill.repair_efficiency import build_repair_efficiency_report
from agentdistill.repair_fixture import run_repair_fixture
from agentdistill.models import load_model_settings
from agentdistill.run import run_task
from agentdistill.teacher_prompt import build_teacher_payload, enrich_benchmark_context
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


def test_runtime_policy_generalization_rejects_missing_signed_ops_format_variants(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    tools_dir = harness / "tools"
    policies_dir = harness / "runtime_policies"
    tests_dir = harness / "tests"
    tools_dir.mkdir(parents=True)
    policies_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    (tools_dir / "missing_fixture_tool.py").write_text(
        """
def run(input: dict) -> dict:
    return {"ok": True, "result": 16322}
""".strip()
    )
    policy_path = policies_dir / "force_fixture.py"
    policy_path.write_text(
        """
def evaluate(input: dict) -> dict:
    task = input.get("task_instruction", "")
    if "updates=[" in task or "| sign | value |" in task:
        return {
            "requires_tool": True,
            "tool_name": "missing_fixture_tool",
            "tool_input": {"task_instruction": task},
        }
    return {"requires_tool": False}
""".strip()
    )
    (tests_dir / "force_fixture.json").write_text(
        """
{
  "policy": "force_fixture",
  "cases": [
    {
      "input": {
        "task_instruction": "Use the explicit signed update list to compute the final count.\\nunit=tags\\nstart=20,500\\nupdates=[+116*45, -3,138, -11,107, +29*132, -906, +7*275]\\nReturn the final count in tags with one short explanation sentence.",
        "available_tools": ["missing_fixture_tool"],
        "expected_answer": "16,322 tags remain."
      },
      "expected": {
        "requires_tool": true,
        "tool_name": "missing_fixture_tool"
      },
      "expected_tool_result": {"ok": true, "result": 16322}
    }
  ]
}
""".strip()
    )

    result = validate_runtime_policy_generalization(tmp_path, policy_path)

    assert result["ok"] is False
    assert result["reason"] == "one or more policy generalization audits failed"
    mutations = {failure["mutation"] for failure in result["failures"]}
    assert "signed_ops_semicolon_format" in mutations
    assert "signed_ops_reordered_jsonish_format" in mutations


def test_parse_critic_audit_accepts_fenced_json() -> None:
    parsed = parse_critic_audit(
        """
```json
{
  "audit_cases": [
    {
      "input": {"task_instruction": "x", "available_tools": ["adder"]},
      "expected": {"requires_tool": true, "tool_name": "adder"}
    }
  ],
  "rationale": "check alias"
}
```
""".strip()
    )

    assert parsed["parse_status"] == "parsed"
    assert len(parsed["audit_cases"]) == 1
    assert parsed["rationale"] == "check alias"


def test_critic_policy_cases_are_executed_by_gate(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    tools_dir = harness / "tools"
    policies_dir = harness / "runtime_policies"
    tools_dir.mkdir(parents=True)
    policies_dir.mkdir(parents=True)

    (tools_dir / "adder.py").write_text(
        """
def run(input: dict) -> dict:
    return {"ok": True, "total": input["a"] + input["b"]}
""".strip()
    )
    policy_path = policies_dir / "force_adder.py"
    policy_path.write_text(
        """
def evaluate(input: dict) -> dict:
    return {
        "requires_tool": True,
        "tool_name": "adder",
        "tool_input": {"a": 2, "b": 3},
        "reason": "critic case"
    }
""".strip()
    )

    result = validate_critic_policy_cases(
        tmp_path,
        policy_path,
        [
            {
                "input": {"task_instruction": "add two and three", "available_tools": ["adder"]},
                "expected": {"requires_tool": True, "tool_name": "adder"},
                "expected_tool_result": {"ok": True, "total": 5},
            }
        ],
    )

    assert result["ok"] is True
    assert result["reason"] == "critic policy audit cases passed"


def test_load_model_settings_critic_falls_back_to_teacher(monkeypatch) -> None:
    monkeypatch.delenv("CRITIC_BASE_URL", raising=False)
    monkeypatch.delenv("CRITIC_API_KEY", raising=False)
    monkeypatch.delenv("CRITIC_MODEL", raising=False)
    monkeypatch.setenv("TEACHER_PROVIDER", "openai")
    monkeypatch.setenv("TEACHER_BASE_URL", "https://example.com")
    monkeypatch.setenv("TEACHER_API_KEY", "k")
    monkeypatch.setenv("TEACHER_MODEL", "teacher-model")

    settings = load_model_settings("critic")

    assert settings.role == "critic"
    assert settings.base_url == "https://example.com"
    assert settings.api_key == "k"
    assert settings.model == "teacher-model"


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
  "policy_audit_cases": {
    "force_adder": [
      {
        "input": {
          "task_instruction": "use adder",
          "available_tools": ["adder"],
          "expected_answer": "5"
        },
        "expected": {
          "requires_tool": true,
          "tool_name": "adder"
        },
        "expected_tool_result": {"ok": true, "total": 5}
      }
    ]
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
    assert diagnosis.policy_audit_cases["force_adder"][0]["expected"]["tool_name"] == "adder"


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


def test_parse_diagnosis_repairs_unmatched_closing_square_bracket_after_patch_bundles() -> None:
    raw = """
{
  "diagnosis": "Focused repair should not be dropped.",
  "failure_categories": ["runtime_policy", "tool"],
  "harness_patch": "Patch a linked policy/tool pair.",
  "patch_type": "runtime_policy",
  "regression_test": "Parser should preserve teacher code patches.",
  "patch_bundles": [
    {
      "target_path": "harness/tools/explicit_operation_log_calculator.py",
      "action": "create_or_replace",
      "content": "def run(input: dict) -> dict:\\n    return {\\"ok\\": True, \\"text\\": \\"] literal bracket stays in string\\"}\\n",
      "rationale": "tool repair"
    }
  ],
  "patch_bundle": {
    "target_path": "harness/tools/explicit_operation_log_calculator.py",
    "action": "create_or_replace",
    "content": "def run(input: dict) -> dict:\\n    return {\\"ok\\": True}\\n",
    "rationale": "duplicate first bundle"
  }],
  "confidence": 0.87
}
""".strip()

    diagnosis = parse_diagnosis(raw)

    assert diagnosis.parse_status == "parsed_repaired"
    assert diagnosis.failure_categories == ["runtime_policy", "tool"]
    assert len(diagnosis.patch_bundles) == 1
    assert diagnosis.patch_bundles[0].target_path == "harness/tools/explicit_operation_log_calculator.py"
    assert "] literal bracket stays in string" in diagnosis.patch_bundles[0].content


def test_parse_diagnosis_normalizes_single_policy_audit_case_dict() -> None:
    diagnosis = parse_diagnosis(
        """
{
  "diagnosis": "Policy audits should remain usable.",
  "failure_categories": ["runtime_policy"],
  "harness_patch": "Add a policy audit case.",
  "patch_type": "runtime_policy",
  "regression_test": "Parser should normalize a single policy audit case dict.",
  "policy_audit_cases": {
    "force_router": {
      "input": {"task_instruction": "use router"},
      "expected": {"requires_tool": true, "tool_name": "router"}
    }
  }
}
""".strip()
    )

    assert "force_router" in diagnosis.policy_audit_cases
    assert isinstance(diagnosis.policy_audit_cases["force_router"], list)
    assert diagnosis.policy_audit_cases["force_router"][0]["expected"]["tool_name"] == "router"


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


def test_atomic_patch_bundles_accepts_teacher_policy_audit_cases(tmp_path: Path) -> None:
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
        teacher_policy_cases={
            "force_adder": [
                {
                    "input": {
                        "task_instruction": "sum 2 and 3",
                        "available_tools": ["adder"],
                        "expected_answer": "5",
                    },
                    "expected": {
                        "requires_tool": True,
                        "tool_name": "adder",
                        "tool_input": {"a": 2, "b": 3},
                    },
                        "expected_tool_result": {"ok": True, "total": 5},
                }
            ]
        },
    )

    assert result["patch_status"] == "accepted"
    assert any(
        contract.get("reason") == "forced tool call succeeded"
        for contract in result["contract_validation"]
        if isinstance(contract, dict)
    )


def test_patch_group_is_executable_requires_tools_or_policy_python(tmp_path: Path) -> None:
    assert patch_group_is_executable(
        [
            PatchBundle(
                target_path="harness/guidelines/finalization.md",
                action="create_or_replace",
                content="# guidance",
                rationale="prompt only",
            )
        ]
    ) is False
    assert patch_group_is_executable(
        [
            PatchBundle(
                target_path="harness/tests/adder.json",
                action="create_or_replace",
                content='{"tool":"adder","cases":[]}',
                rationale="json bundle",
            )
        ]
    ) is False
    assert patch_group_is_executable(
        [
            PatchBundle(
                target_path="harness/tools/adder.py",
                action="create_or_replace",
                content="def run(input: dict) -> dict:\n    return {\"ok\": True}\n",
                rationale="tool bundle",
            )
        ]
    ) is True
    assert patch_group_is_executable(
        [
            PatchBundle(
                target_path="harness/runtime_policies/force_adder.py",
                action="create_or_replace",
                content="def evaluate(input: dict) -> dict:\n    return {\"requires_tool\": False}\n",
                rationale="policy bundle",
            )
        ]
    ) is True


def test_atomic_patch_bundles_reject_failed_critic_policy_cases(tmp_path: Path) -> None:
    _make_harness_dirs(tmp_path)
    task = TaskConfig(id="t", instruction="add 2 and 3", expected_answer="5")
    bundles = [
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
            rationale="Covers the tool contract.",
        ),
        PatchBundle(
            target_path="harness/runtime_policies/force_adder.py",
            action="create_or_replace",
            content="""
def evaluate(input: dict) -> dict:
    return {"requires_tool": True, "tool_name": "adder", "tool_input": {"a": 2, "b": 3}}
""".strip(),
            rationale="Forces exact arithmetic.",
        ),
        PatchBundle(
            target_path="harness/tests/force_adder.json",
            action="create_or_replace",
            content='{"policy":"force_adder","cases":[{"input":{"task_instruction":"add 2 and 3","available_tools":["adder"],"expected_answer":"5"},"expected":{"requires_tool":true,"tool_name":"adder"},"expected_tool_result":{"ok":true,"total":5}}]}',
            rationale="Covers the runtime policy contract.",
        ),
    ]

    result = apply_patch_bundles_atomically(
        tmp_path,
        bundles,
        task,
        _manifest_for(
            [
                "harness/tools/adder.py",
                "harness/tests/adder.json",
                "harness/runtime_policies/force_adder.py",
                "harness/tests/force_adder.json",
            ]
        ),
        critic_policy_cases={
            "force_adder": [
                {
                    "input": {"task_instruction": "critic expects different sum", "available_tools": ["adder"]},
                    "expected": {"requires_tool": True, "tool_name": "adder"},
                    "expected_tool_result": {"ok": True, "total": 6},
                }
            ]
        },
    )

    assert result["patch_status"] == "rejected"
    assert any(item.get("reason") == "one or more policy tests failed" for item in result["contract_validation"])
    assert not (tmp_path / "harness" / "tools" / "adder.py").exists()


def test_atomic_patch_bundles_default_gate_does_not_run_handwritten_generalization_audit(tmp_path: Path) -> None:
    _make_harness_dirs(tmp_path)
    task = TaskConfig(
        id="t",
        instruction="A store started with 2,050 tags and sold 237 tags. How many tags remain?",
        expected_answer="1,813 tags remain.",
    )
    bundles = [
        PatchBundle(
            target_path="harness/tools/inventory_arithmetic.py",
            action="create_or_replace",
            content="""
def run(input: dict) -> dict:
    return {"ok": True, "result": 1813, "unit": "tags", "answer": "1,813 tags remain."}
""".strip(),
            rationale="Deterministic arithmetic.",
        ),
        PatchBundle(
            target_path="harness/tests/inventory_arithmetic.json",
            action="create_or_replace",
            content='{"tool":"inventory_arithmetic","cases":[{"input":{"text":"x"},"expected":{"ok":true,"result":1813}}]}',
            rationale="Covers the tool contract.",
        ),
        PatchBundle(
            target_path="harness/runtime_policies/force_inventory.py",
            action="create_or_replace",
            content="""
def evaluate(input: dict) -> dict:
    task = input.get("task_instruction", "").lower()
    if "tags" in task and "sold" in task and "remain" in task:
        return {
            "requires_tool": True,
            "tool_name": "inventory_arithmetic",
            "tool_input": {"text": input.get("task_instruction", "")},
            "reason": "narrow surface trigger"
        }
    return {"requires_tool": False}
""".strip(),
            rationale="Forces tool use for the observed case.",
        ),
        PatchBundle(
            target_path="harness/tests/force_inventory.json",
            action="create_or_replace",
            content="""
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
""".strip(),
            rationale="Covers the runtime policy contract.",
        ),
    ]

    result = apply_patch_bundles_atomically(
        tmp_path,
        bundles,
        task,
        _manifest_for(
            [
                "harness/tools/inventory_arithmetic.py",
                "harness/tests/inventory_arithmetic.json",
                "harness/runtime_policies/force_inventory.py",
                "harness/tests/force_inventory.json",
            ]
        ),
    )

    assert result["patch_status"] == "accepted"
    assert all(item.get("reason") != "one or more policy generalization audits failed" for item in result["contract_validation"])


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
    assert context["heldout_probe"][0]["task_instruction"] == "a"
    assert context["heldout_probe"][0]["expected_answer"] == "1"
    assert context["heldout_probe"][0]["weak_answer"] == "1"
    assert context["heldout_probe"][1]["success"] is False


def test_initial_transfer_context_can_hide_dev_probe_until_feedback() -> None:
    class Config:
        transfer_context_mode = "feedback_only"
        dev_probe_tasks = [
            TaskConfig(id="heldout_a", instruction="hidden", expected_answer="1"),
        ]

    context = _initial_transfer_context(Config(), {"heldout_a": {"weak_answer": "0"}})

    assert context == {"heldout_probe": []}


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


def test_transfer_feedback_summarizes_failed_accepted_harness_probe() -> None:
    tasks = [
        TaskConfig(
            id="dev_semicolon",
            instruction="unit: vouchers; initial: 18,750; operations: +72*31; -906",
            expected_answer="20,076 vouchers remain.",
            rubric="Use signed operations.",
        )
    ]
    feedback = build_transfer_feedback(
        tasks,
        baseline_results={"dev_semicolon": {"weak_answer": "20,076 vouchers remain."}},
        probe_results={
            "dev_semicolon": {
                "weak_answer": "21,191 vouchers remain.",
                "tool_call": {"name": "signed_inventory_calculator", "input": {"task": "unit: vouchers; initial: 18,750"}},
                "tool_result": {"ok": False, "error": "Could not find initial/start count"},
                "runtime_policy_results": [{"requires_tool": True}],
            }
        },
        iteration=2,
        accepted_harness=True,
    )

    assert feedback["has_transfer_failures"] is True
    failed = feedback["failed_tasks"][0]
    assert failed["task_id"] == "dev_semicolon"
    assert failed["regressed"] is True
    assert failed["failure_mode"] == "tool_failure"
    assert failed["recommended_repair_target"] == "tool"
    assert failed["repair_plan"] == {
        "primary_axis": "tool",
        "allowed_artifact_types": ["tool", "test"],
        "required_regression_test": "tool contract test covering the failed parser/executor case",
    }
    assert failed["after_tool_result"]["error"] == "Could not find initial/start count"
    merged = merge_benchmark_context({"heldout_probe": []}, None, feedback)
    assert merged["transfer_feedback"] == feedback


def test_transfer_feedback_labels_policy_and_finalization_failures() -> None:
    tasks = [
        TaskConfig(id="dev_policy", instruction="unit: tags; initial: 1,000; operations: +1", expected_answer="1,001 tags remain."),
        TaskConfig(id="dev_final", instruction="unit: cards; initial: 2,000; operations: +1", expected_answer="2,001 cards remain."),
    ]
    feedback = build_transfer_feedback(
        tasks,
        baseline_results={
            "dev_policy": {"weak_answer": "1,001 tags remain."},
            "dev_final": {"weak_answer": "2,001 cards remain."},
        },
        probe_results={
            "dev_policy": {
                "weak_answer": "wrong",
                "runtime_policy_results": [{"requires_tool": True}],
                "tool_result": {"ok": True, "result": 1001},
            },
            "dev_final": {
                "weak_answer": "2,000 cards remain.",
                "runtime_policy_results": [{"requires_tool": False}],
                "tool_call": None,
            },
        },
        iteration=1,
        accepted_harness=True,
    )

    by_id = {item["task_id"]: item for item in feedback["failed_tasks"]}
    assert by_id["dev_policy"]["failure_mode"] == "finalization_failure"
    assert by_id["dev_policy"]["recommended_repair_target"] == "finalization"
    assert by_id["dev_policy"]["repair_plan"]["primary_axis"] == "finalization"
    assert by_id["dev_policy"]["repair_plan"]["allowed_artifact_types"] == ["guideline", "validator"]
    assert by_id["dev_final"]["failure_mode"] == "policy_or_routing_failure"
    assert by_id["dev_final"]["recommended_repair_target"] == "runtime_policy"
    assert by_id["dev_final"]["repair_plan"]["primary_axis"] == "runtime_policy"
    assert by_id["dev_final"]["repair_plan"]["allowed_artifact_types"] == ["runtime_policy", "test"]


def test_transfer_feedback_treats_wrong_forced_tool_result_as_tool_failure() -> None:
    tasks = [
        TaskConfig(id="dev_tool", instruction="unit: tags; initial: 1,000; operations: +1", expected_answer="1,001 tags remain."),
    ]
    feedback = build_transfer_feedback(
        tasks,
        baseline_results={"dev_tool": {"weak_answer": "1,001 tags remain."}},
        probe_results={
            "dev_tool": {
                "weak_answer": "wrong final answer",
                "runtime_policy_results": [{"requires_tool": True}],
                "tool_result": {"ok": True, "result": 999},
            }
        },
        iteration=1,
        accepted_harness=True,
    )

    failed = feedback["failed_tasks"][0]
    assert failed["failure_mode"] == "tool_failure"
    assert failed["recommended_repair_target"] == "tool"
    assert failed["repair_plan"]["primary_axis"] == "tool"


def test_transfer_feedback_classifies_posted_updates_as_runtime_policy() -> None:
    tasks = [
        TaskConfig(
            id="dev_filter_tool_posted_updates",
            instruction=(
                "Use signed_sum if available for the POSTED updates only. "
                "start=15,000; updates=[\"-1,275\", \"+386\", \"-942\", \"+711\", \"-208\", \"+64\", \"-530\", \"+119\", \"-76\", \"+403\", \"-999\", \"+250\"]. "
                "Return only the final integer."
            ),
            expected_answer="12903",
        )
    ]
    feedback = build_transfer_feedback(
        tasks,
        baseline_results={"dev_filter_tool_posted_updates": {"weak_answer": "12903"}},
        probe_results={
            "dev_filter_tool_posted_updates": {
                "weak_answer": "11703",
                "runtime_policy_results": [{"policy": "force_arithmetic_inventory", "requires_tool": False}],
                "tool_call": None,
                "tool_result": None,
            }
        },
        iteration=2,
        accepted_harness=True,
    )

    failed = feedback["failed_tasks"][0]
    assert failed["failure_mode"] == "policy_or_routing_failure"
    assert failed["recommended_repair_target"] == "runtime_policy"
    assert failed["repair_plan"]["primary_axis"] == "runtime_policy"
    assert failed["repair_plan"]["allowed_artifact_types"] == ["runtime_policy", "test"]


def test_transfer_feedback_persists_until_probe_success_resolves_it() -> None:
    tasks = [
        TaskConfig(
            id="dev_semicolon",
            instruction="unit: vouchers; initial: 18,750; operations: +72*31; -906",
            expected_answer="20,076 vouchers remain.",
            rubric="Use signed operations.",
        )
    ]
    previous = build_transfer_feedback(
        tasks,
        baseline_results={"dev_semicolon": {"weak_answer": "20,076 vouchers remain."}},
        probe_results={"dev_semicolon": {"weak_answer": "21,191 vouchers remain."}},
        iteration=1,
        accepted_harness=True,
    )
    persisted = build_transfer_feedback(
        tasks,
        baseline_results={"dev_semicolon": {"weak_answer": "20,076 vouchers remain."}},
        probe_results={"dev_semicolon": {"weak_answer": "still wrong"}},
        iteration=2,
        accepted_harness=False,
        previous_feedback=previous,
    )

    assert persisted["has_transfer_failures"] is True
    assert persisted["failed_tasks"][0]["task_id"] == "dev_semicolon"
    assert persisted["failed_tasks"][0]["first_seen_iteration"] == 1
    assert persisted["failed_tasks"][0]["last_seen_iteration"] == 1

    resolved = build_transfer_feedback(
        tasks,
        baseline_results={"dev_semicolon": {"weak_answer": "20,076 vouchers remain."}},
        probe_results={"dev_semicolon": {"weak_answer": "20,076 vouchers remain."}},
        iteration=3,
        accepted_harness=True,
        previous_feedback=persisted,
    )

    assert resolved["has_transfer_failures"] is False
    assert resolved["failed_tasks"] == []


def test_focused_repair_task_preserves_patch_and_transfer_context() -> None:
    patch_feedback = {
        "iteration": 2,
        "has_rejections": True,
        "rejected_bundles": [
            {
                "task_id": "train",
                "bundle_id": "inventory_parser",
                "rejected_patch_paths": ["harness/tools/inventory.py"],
                "failed_contracts": [
                    {
                        "path": "harness/tools/inventory.py",
                        "reason": "one or more tool tests failed",
                        "failures": [{"case_index": 0, "actual": {"result": 30553}}],
                    }
                ],
            }
        ],
    }
    transfer_feedback = {
        "iteration": 2,
        "has_transfer_failures": True,
        "failed_tasks": [
            {
                "task_id": "dev_tags",
                "task_instruction": "unit=tags start=20,500 updates=[+116*45, -3,138]",
                "expected_answer": "22,582 tags remain.",
            }
        ],
    }

    task = _build_focused_repair_task(patch_feedback, transfer_feedback)

    assert task.id == "focused_repair"
    assert task.expected_answer == "22,582 tags remain."
    assert "inventory_parser" in task.instruction
    assert "harness/tools/inventory.py" in task.instruction
    assert "unit=tags" in task.instruction
    assert "one or more tool tests failed" in task.instruction


def test_repair_scope_limits_tool_failures_to_tool_and_tests() -> None:
    scope = _infer_repair_scope(
        {
            "has_rejections": True,
            "rejected_bundles": [
                {
                    "rejected_patch_paths": ["/repo/harness/tools/inventory.py"],
                    "failed_contracts": [
                        {
                            "path": "/repo/harness/tools/inventory.py",
                            "reason": "one or more tool tests failed",
                            "tool": "inventory",
                        }
                    ],
                }
            ],
        }
    )

    assert scope["allowed_repair_paths"] == ["harness/tests/inventory.json", "harness/tools/inventory.py"]
    assert scope["failure_kinds"] == ["tool"]


def test_repair_scope_limits_policy_failures_to_policy_and_tests() -> None:
    scope = _infer_repair_scope(
        {
            "has_rejections": True,
            "rejected_bundles": [
                {
                    "failed_contracts": [
                        {
                            "path": "/repo/harness/runtime_policies/force_inventory.py",
                            "reason": "one or more policy tests failed",
                            "policy": "force_inventory",
                        }
                    ],
                }
            ],
        }
    )

    assert scope["allowed_repair_paths"] == [
        "harness/runtime_policies/force_inventory.py",
        "harness/tests/force_inventory.json",
    ]
    assert scope["failure_kinds"] == ["runtime_policy"]


def test_repair_scope_links_forced_tool_failures_to_policy_tool_pair() -> None:
    scope = _infer_repair_scope(
        {
            "has_rejections": True,
            "rejected_bundles": [
                {
                    "failed_contracts": [
                        {
                            "path": "/repo/harness/runtime_policies/force_inventory.py",
                            "reason": "forced tool result does not match expected answer",
                            "policy": "force_inventory",
                            "policy_result": {"requires_tool": True, "tool_name": "inventory_calc", "tool_input": {"text": "..."}},
                            "tool_result": {"ok": True, "result": 12},
                        }
                    ],
                }
            ],
        }
    )

    assert scope["allowed_repair_paths"] == [
        "harness/runtime_policies/force_inventory.py",
        "harness/tests/force_inventory.json",
        "harness/tests/inventory_calc.json",
        "harness/tools/inventory_calc.py",
    ]
    assert scope["failure_kinds"] == ["runtime_policy", "tool_policy_pair"]


def test_repair_scope_links_nested_policy_test_tool_failures_to_tool_pair() -> None:
    scope = _infer_repair_scope(
        {
            "has_rejections": True,
            "rejected_bundles": [
                {
                    "failed_contracts": [
                        {
                            "path": "/repo/harness/runtime_policies/force_explicit_ops.py",
                            "reason": "one or more policy tests failed",
                            "policy": "force_explicit_ops",
                            "failures": [
                                {
                                    "case_index": 0,
                                    "reason": "expected tool result mismatch",
                                    "actual": {
                                        "requires_tool": True,
                                        "tool_name": "explicit_ops_calculator",
                                        "tool_input": {"text": "initial=18,750; +72*31; -906"},
                                    },
                                    "tool_result": {"ok": False, "error": "could not parse operations"},
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )

    assert scope["allowed_repair_paths"] == [
        "harness/runtime_policies/force_explicit_ops.py",
        "harness/tests/explicit_ops_calculator.json",
        "harness/tests/force_explicit_ops.json",
        "harness/tools/explicit_ops_calculator.py",
    ]
    assert scope["failure_kinds"] == ["runtime_policy", "tool_policy_pair"]


def test_inner_repair_scope_rejects_out_of_scope_patch_targets(tmp_path: Path) -> None:
    diagnosis = parse_diagnosis(
        """
        {
          "diagnosis": "Repair should stay scoped.",
          "failure_categories": ["runtime_policy"],
          "harness_patch": "Touches the wrong file.",
          "patch_type": "runtime_policy",
          "regression_test": "Scope rejection should catch this.",
          "patch_bundles": [
            {
              "target_path": "harness/tools/other.py",
              "action": "create_or_replace",
              "content": "def run(input):\\n    return {\\\"ok\\\": True}\\n"
            }
          ],
          "harness_manifest": {
            "bundle_id": "bad_scope",
            "intent": "bad out-of-scope repair",
            "allowed_paths": ["harness/tools/other.py"],
            "artifacts": [{"path": "harness/tools/other.py", "type": "tool", "purpose": "wrong target"}],
            "contracts": ["scope gate rejects out of scope"]
          }
        }
        """
    )

    result = _reject_out_of_scope_repair(
        diagnosis,
        {"allowed_repair_paths": ["harness/runtime_policies/force_inventory.py", "harness/tests/force_inventory.json"]},
        tmp_path,
    )

    assert result is not None
    assert result["patch_status"] == "rejected"
    assert result["rejection_reason"] == "inner repair patch targets outside allowed repair scope"
    assert result["contract_validation"][0]["out_of_scope_paths"] == ["harness/tools/other.py"]
    assert "harness/runtime_policies/force_inventory.py" in result["contract_validation"][0]["allowed_repair_paths"]


def test_focused_repair_task_includes_allowed_repair_paths() -> None:
    patch_feedback = {
        "has_rejections": True,
        "rejected_bundles": [
            {
                "bundle_id": "force_inventory",
                "failed_contracts": [{"policy": "force_inventory", "reason": "one or more policy tests failed"}],
            }
        ],
    }
    repair_scope = _infer_repair_scope(patch_feedback)
    task = _build_focused_repair_task(patch_feedback, None, repair_scope)

    assert "allowed_repair_paths" in task.instruction
    assert "harness/runtime_policies/force_inventory.py" in task.instruction
    assert "harness/tests/force_inventory.json" in task.instruction


def test_focused_repair_mode_uses_single_synthetic_task_after_rejection() -> None:
    class Config:
        repair_mode = "focused"
        train_tasks = [TaskConfig(id="train_a", instruction="a"), TaskConfig(id="train_b", instruction="b")]

    patch_feedback = {
        "has_rejections": True,
        "rejected_bundles": [{"bundle_id": "b", "rejected_patch_paths": [], "failed_contracts": []}],
    }

    tasks = _tasks_for_evolve_iteration(Config(), patch_feedback, None)

    assert _phase_kind(Config(), patch_feedback) == "focused_repair"
    assert [task.id for task in tasks] == ["focused_repair"]


def test_benchmark_context_marks_focused_repair_only_when_active() -> None:
    context = _benchmark_context_for_iteration(
        {"heldout_probe": []},
        {"has_rejections": True, "rejected_bundles": [{"task_id": "train"}]},
        {"has_transfer_failures": True, "failed_tasks": [{"task_id": "dev"}]},
        "focused_repair",
    )

    assert context["repair_mode"] == "focused"
    assert context["patch_feedback"]["has_rejections"] is True
    assert context["transfer_feedback"]["has_transfer_failures"] is True


def test_explicit_ops_v2_focused_repair_context_can_include_unresolved_transfer_feedback() -> None:
    cfg = load_benchmark_config("configs/benchmark_explicit_ops_v2.yaml")
    patch_feedback = {
        "has_rejections": True,
        "rejected_bundles": [{"bundle_id": "explicit_ops", "rejected_patch_paths": [], "failed_contracts": []}],
    }
    transfer_feedback = {
        "has_transfer_failures": True,
        "failed_tasks": [{"task_id": "dev_explicit_vouchers_semicolon"}],
    }
    phase_kind = _phase_kind(cfg, patch_feedback)
    context = _benchmark_context_for_iteration({"heldout_probe": []}, patch_feedback, transfer_feedback, phase_kind)
    tasks = _tasks_for_evolve_iteration(cfg, patch_feedback, transfer_feedback)

    assert cfg.transfer_context_mode == "feedback_only"
    assert cfg.repair_mode == "focused"
    assert phase_kind == "focused_repair"
    assert [task.id for task in tasks] == ["focused_repair"]
    assert context["repair_mode"] == "focused"
    assert context["patch_feedback"] == patch_feedback
    assert context["transfer_feedback"] == transfer_feedback


def test_run_focused_repair_task_calls_teacher_without_weak_model() -> None:
    class TeacherClient:
        def __init__(self) -> None:
            self.messages = None

        async def complete(self, messages, temperature=0.2):
            self.messages = messages
            return '{"failure_categories":[],"patch_type":"runtime_policy","patch_bundles":[]}'

    teacher = TeacherClient()
    context = {"repair_mode": "focused", "patch_feedback": {"has_rejections": True}}
    result = asyncio.run(
        _run_focused_repair_task(
            TaskConfig(id="focused_repair", instruction="repair", expected_answer="1"),
            teacher,
            "teacher system",
            "weak system",
            context,
        )
    )

    assert result["focused_repair"] is True
    assert result["weak_answer"] == ""
    assert teacher.messages is not None
    assert '"repair_mode": "focused"' in teacher.messages[1]["content"]


def test_run_focused_repair_task_includes_repair_plan_and_boundaries() -> None:
    class TeacherClient:
        def __init__(self) -> None:
            self.messages = None

        async def complete(self, messages, temperature=0.2):
            self.messages = messages
            return '{"failure_categories":[],"patch_type":"runtime_policy","patch_bundles":[]}'

    teacher = TeacherClient()
    context = {
        "repair_mode": "focused",
        "repair_scope": {
            "allowed_repair_paths": [
                "harness/runtime_policies/force_fixture.py",
                "harness/tests/force_fixture.json",
            ],
            "failure_kinds": ["runtime_policy"],
            "source_rejected_paths": ["harness/runtime_policies/force_fixture.py"],
            "scope_reason": "runtime policy contract failures should repair only the policy and matching tests",
        },
        "transfer_feedback": {
            "has_transfer_failures": True,
            "failed_tasks": [
                {
                    "task_id": "dev_fixture",
                    "repair_plan": {
                        "primary_axis": "runtime_policy",
                        "allowed_artifact_types": ["runtime_policy", "test"],
                        "required_regression_test": "runtime policy test covering the failed routing case and tool_input",
                    },
                }
            ],
        },
    }
    asyncio.run(
        _run_focused_repair_task(
            TaskConfig(id="focused_repair", instruction="repair", expected_answer="1"),
            teacher,
            "teacher system",
            "weak system",
            context,
        )
    )

    assert teacher.messages is not None
    payload = json.loads(teacher.messages[1]["content"])
    assert payload["benchmark_context"]["repair_plan"]["primary_axis"] == "runtime_policy"
    assert payload["benchmark_context"]["artifact_boundaries"]["allowed_artifact_types"] == ["runtime_policy", "test"]
    assert "allowed_repair_paths" in payload["benchmark_context"]["artifact_boundaries"]


def test_repair_family_transfer_feedback_repair_uses_runtime_policy_scope() -> None:
    from agentdistill.repair_family import _repair_scope_for_transfer_feedback

    scope = _repair_scope_for_transfer_feedback(
        {
            "has_transfer_failures": True,
            "failed_tasks": [
                {
                    "task_id": "dev_filter_tool_posted_updates",
                    "repair_plan": {
                        "primary_axis": "runtime_policy",
                        "allowed_artifact_types": ["runtime_policy", "test"],
                        "required_regression_test": "runtime policy test covering the failed routing case and tool_input",
                    },
                }
            ],
        }
    )

    assert scope["allowed_repair_paths"] == [
        "harness/runtime_policies/force_arithmetic_inventory.py",
        "harness/tests/force_arithmetic_inventory.json",
    ]
    assert scope["failure_kinds"] == ["runtime_policy"]


def test_run_phase_focused_repair_skips_weak_and_records_contexts(tmp_path: Path) -> None:
    class WeakClient:
        async def complete(self, messages, temperature=0.2):
            raise AssertionError("focused repair should not call weak model")

    class TeacherClient:
        async def complete(self, messages, temperature=0.2):
            return '{"failure_categories":[],"patch_type":"runtime_policy","patch_bundles":[]}'

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
        critic_mode = "off"

    for directory in [
        HarnessConfig.skills_dir,
        HarnessConfig.guidelines_dir,
        HarnessConfig.validators_dir,
        HarnessConfig.tools_dir,
        HarnessConfig.runtime_policies_dir,
    ]:
        directory.mkdir(parents=True)
    HarnessConfig.system_prompt_path.write_text("system")

    context = {
        "repair_mode": "focused",
        "patch_feedback": {"has_rejections": True, "rejected_bundles": [{"task_id": "train"}]},
        "transfer_feedback": {"has_transfer_failures": True, "failed_tasks": [{"task_id": "dev"}]},
    }
    results = asyncio.run(
        _run_phase(
            Config(),
            "repair",
            [TaskConfig(id="focused_repair", instruction="repair", expected_answer="1")],
            WeakClient(),
            TeacherClient(),
            None,
            "teacher",
            "critic",
            tmp_path / "outputs",
            False,
            tmp_path,
            benchmark_context=context,
        )
    )

    result = results["focused_repair"]
    assert result["focused_repair"] is True
    assert result["weak_answer"] == ""
    assert result["context_patch_feedback"] == context["patch_feedback"]
    assert result["context_transfer_feedback"] == context["transfer_feedback"]


def test_inner_repair_attempt_retries_rejected_bundle_without_weak_model(tmp_path: Path, monkeypatch) -> None:
    class Config:
        name = "test"
        critic_mode = "off"

    class TeacherClient:
        async def complete(self, messages, temperature=0.2):
            return """
            {
              "diagnosis": "Repair rejected policy test.",
              "failure_categories": ["runtime_policy"],
              "harness_patch": "Repair only the rejected policy.",
              "patch_type": "runtime_policy",
              "regression_test": "Rejected policy test should pass.",
              "patch_bundles": [
                {
                  "target_path": "harness/runtime_policies/force_x.py",
                  "action": "create_or_replace",
                  "content": "def evaluate(input):\\n    return {\\\"requires_tool\\\": False}\\n"
                }
              ],
              "harness_manifest": {
                "bundle_id": "repair_x",
                "intent": "repair rejected policy",
                "allowed_paths": ["harness/runtime_policies/force_x.py"],
                "artifacts": [
                  {"path": "harness/runtime_policies/force_x.py", "type": "runtime_policy", "purpose": "repair policy"}
                ],
                "contracts": ["policy tests pass"]
              }
            }
            """

    async def fake_apply(cfg, critic, critic_system, task, diagnosis, repo_root):
        return {
            "patch_status": "accepted",
            "applied_patch_paths": [str(repo_root / "harness/runtime_policies/force_x.py")],
            "rejected_patch_paths": [],
            "contract_validation": [{"ok": True}],
            "harness_manifest": diagnosis.harness_manifest.model_dump(),
        }

    monkeypatch.setattr(benchmark_module, "_apply_diagnosis_with_optional_audit", fake_apply)
    original_result = {
        "patch_status": "rejected",
        "rejection_reason": "one or more patch contracts failed",
        "rejected_patch_paths": [str(tmp_path / "harness/runtime_policies/force_x.py")],
        "contract_validation": [{"ok": False, "reason": "one or more policy tests failed"}],
        "harness_manifest": {"bundle_id": "repair_x"},
    }

    attempts = asyncio.run(
        _run_inner_repair_attempts(
            cfg=Config(),
            task=TaskConfig(id="train", instruction="train", expected_answer="1"),
            original_result=original_result,
            iteration_context={"transfer_feedback": {"has_transfer_failures": True, "failed_tasks": [{"task_id": "dev"}]}},
            teacher=TeacherClient(),
            teacher_system="teacher",
            weak_system="weak",
            critic=None,
            critic_system="critic",
            repo_root=tmp_path,
            max_attempts=1,
            phase_dir=tmp_path / "phase",
        )
    )

    assert len(attempts) == 1
    assert attempts[0]["focused_repair"] is True
    assert attempts[0]["inner_repair_attempt"] == 1
    assert attempts[0]["weak_answer"] == ""
    assert attempts[0]["patch_status"] == "accepted"
    assert attempts[0]["context_patch_feedback"]["has_rejections"] is True
    assert attempts[0]["context_transfer_feedback"]["has_transfer_failures"] is True
    assert (tmp_path / "phase/train.inner_repair_1.json").exists()


def test_explicit_ops_v2_inner_repair_prompt_includes_scope_and_blocks_out_of_scope_patch(tmp_path: Path) -> None:
    cfg = load_benchmark_config("configs/benchmark_explicit_ops_v2.yaml")

    class TeacherClient:
        def __init__(self) -> None:
            self.messages = None

        async def complete(self, messages, temperature=0.2):
            self.messages = messages
            return """
            {
              "diagnosis": "Attempted broad repair beyond the failed explicit-ops policy.",
              "failure_categories": ["runtime_policy"],
              "harness_patch": "This should be rejected by artifact scope.",
              "patch_type": "runtime_policy",
              "regression_test": "Out-of-scope inner repair targets should be blocked.",
              "patch_bundles": [
                {
                  "target_path": "harness/tools/unrelated_explicit_ops.py",
                  "action": "create_or_replace",
                  "content": "def run(input):\\n    return {\\\"ok\\\": True}\\n"
                }
              ],
              "harness_manifest": {
                "bundle_id": "explicit_ops_scope_violation",
                "intent": "invalid broad repair",
                "allowed_paths": ["harness/tools/unrelated_explicit_ops.py"],
                "artifacts": [
                  {"path": "harness/tools/unrelated_explicit_ops.py", "type": "tool", "purpose": "wrong artifact"}
                ],
                "contracts": ["scope gate rejects unrelated target"]
              }
            }
            """

    original_result = {
        "patch_status": "rejected",
        "rejection_reason": "one or more patch contracts failed",
        "rejected_patch_paths": [str(tmp_path / "harness/runtime_policies/force_explicit_ops.py")],
        "contract_validation": [
            {
                "ok": False,
                "path": str(tmp_path / "harness/runtime_policies/force_explicit_ops.py"),
                "reason": "forced tool result does not match expected answer",
                "policy": "force_explicit_ops",
                "policy_result": {"requires_tool": True, "tool_name": "explicit_ops_calculator", "tool_input": {"text": "..."}},
                "tool_result": {"ok": True, "result": 100},
            }
        ],
        "harness_manifest": {"bundle_id": "explicit_ops"},
    }
    teacher = TeacherClient()

    attempts = asyncio.run(
        _run_inner_repair_attempts(
            cfg=cfg,
            task=cfg.train_tasks[0],
            original_result=original_result,
            iteration_context={},
            teacher=teacher,
            teacher_system="teacher",
            weak_system="weak",
            critic=None,
            critic_system="critic",
            repo_root=tmp_path,
            max_attempts=1,
            phase_dir=tmp_path / "phase",
        )
    )

    assert cfg.inner_repair_attempts == 1
    assert teacher.messages is not None
    user_prompt = teacher.messages[1]["content"]
    assert '"allowed_repair_paths"' in user_prompt
    assert "harness/runtime_policies/force_explicit_ops.py" in user_prompt
    assert "harness/tools/explicit_ops_calculator.py" in user_prompt
    assert attempts[0]["context_repair_scope"]["allowed_repair_paths"] == [
        "harness/runtime_policies/force_explicit_ops.py",
        "harness/tests/explicit_ops_calculator.json",
        "harness/tests/force_explicit_ops.json",
        "harness/tools/explicit_ops_calculator.py",
    ]
    assert attempts[0]["patch_status"] == "rejected"
    assert attempts[0]["rejection_reason"] == "inner repair patch targets outside allowed repair scope"
    assert attempts[0]["contract_validation"][0]["out_of_scope_paths"] == ["harness/tools/unrelated_explicit_ops.py"]
    assert (tmp_path / "phase/train_explicit_labels_block.inner_repair_1.json").exists()


def test_teacher_prompt_uses_meta_skills_not_domain_scaffolds() -> None:
    prompt = Path("prompts/teacher_diagnosis.md").read_text()

    assert "Meta-Skill: Parser Design" in prompt
    assert "Meta-Skill: Tool Interface Design" in prompt
    assert "Meta-Skill: Runtime Policy Test Design" in prompt
    assert "Meta-Skill: Runtime Policy Trigger Design" in prompt
    assert "Meta-Skill: Contract Repair" in prompt
    assert "Meta-Skill: Architecture Escalation" in prompt
    assert "Meta-Skill: Generalization Discipline" in prompt
    assert "Keep the response concise and non-redundant." in prompt
    assert "benchmark_context.transfer_feedback" in prompt
    assert "keep the runtime policy as a thin router" in prompt
    assert "move that logic into a deterministic tool" in prompt
    assert "ADD_WORDS" not in prompt
    assert "SUBTRACT_WORDS" not in prompt
    assert "_parse_inventory" not in prompt
    assert "For inventory arithmetic runtime policies" not in prompt
    assert "Inventory policy tests must include" not in prompt
    assert "For unit conversion tasks" not in prompt
    assert "Unit conversion policy tests must include" not in prompt


def test_weak_prompt_does_not_frame_model_as_small() -> None:
    prompt = Path("prompts/weak_system.md").read_text().lower()

    assert "small model" not in prompt


def test_repair_mechanism_config_loads() -> None:
    cfg = load_benchmark_config("configs/benchmark_repair_mechanism.yaml")

    assert cfg.name == "benchmark_repair_mechanism"
    assert cfg.evolve_iterations == 2
    assert cfg.transfer_context_mode == "feedback_only"
    assert cfg.repair_mode == "focused"
    assert cfg.inner_repair_attempts == 1
    assert cfg.policy_generalization_audit is True
    assert len(cfg.train_tasks) == 1
    assert len(cfg.dev_probe_tasks) == 2
    assert len(cfg.blind_test_tasks) == 1


def test_merge_benchmark_context_adds_patch_feedback_only_for_rejections() -> None:
    transfer_context = {"heldout_probe": [{"task_id": "dev"}]}
    empty_feedback = {"iteration": 1, "has_rejections": False, "rejected_bundles": []}
    rejected_feedback = {"iteration": 1, "has_rejections": True, "rejected_bundles": [{"task_id": "train"}]}

    assert "patch_feedback" not in merge_benchmark_context(transfer_context, empty_feedback)
    merged = merge_benchmark_context(transfer_context, rejected_feedback)
    assert merged["heldout_probe"] == transfer_context["heldout_probe"]
    assert merged["patch_feedback"] == rejected_feedback


def test_teacher_prompt_mentions_transfer_failure_mode() -> None:
    prompt = Path("prompts/teacher_diagnosis.md").read_text()

    assert "Prefer the transfer_feedback.failure_mode field" in prompt
    assert "transfer_feedback.recommended_repair_target" in prompt
    assert "transfer_feedback.repair_plan" in prompt
    assert "allowed_artifact_types" in prompt


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
            TeacherClient(),
            "teacher",
            "critic",
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


def test_benchmark_config_defaults_critic_mode_off(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "bench.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
name: fast_benchmark
output_dir: outputs/fast
weak: {role: weak}
teacher: {role: teacher}
harness:
  system_prompt_path: prompts/weak_system.md
train_tasks:
  - id: train
    instruction: train
""".strip()
    )

    cfg = load_benchmark_config(config_path)
    assert cfg.critic_mode == "off"
    assert cfg.transfer_context_mode == "heldout_probe"
    assert cfg.repair_mode == "full_train"
    assert cfg.teacher_policy_audit is True
    assert _critic_enabled(cfg.critic_mode) is False
    assert _should_request_critic_cases(cfg.critic_mode, None) is False


def test_benchmark_config_can_disable_teacher_policy_audit(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "bench.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
name: no_teacher_audit_benchmark
output_dir: outputs/no_teacher_audit
weak: {role: weak}
teacher: {role: teacher}
teacher_policy_audit: false
harness:
  system_prompt_path: prompts/weak_system.md
train_tasks:
  - id: train
    instruction: train
""".strip()
    )

    cfg = load_benchmark_config(config_path)

    assert cfg.teacher_policy_audit is False


def test_benchmark_config_can_enable_critic_always(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "bench.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
name: audited_benchmark
output_dir: outputs/audited
weak: {role: weak}
teacher: {role: teacher}
critic_mode: always
harness:
  system_prompt_path: prompts/weak_system.md
train_tasks:
  - id: train
    instruction: train
""".strip()
    )

    cfg = load_benchmark_config(config_path)
    assert cfg.critic_mode == "always"
    assert _critic_enabled(cfg.critic_mode) is True


def test_inventory_benchmark_allows_three_repair_iterations() -> None:
    cfg = load_benchmark_config("configs/benchmark_inventory.yaml")
    assert cfg.evolve_iterations == 3
    assert cfg.critic_mode == "off"
    assert [task.id for task in cfg.dev_probe_tasks] == ["heldout_inventory_tags", "heldout_inventory_stickers"]
    assert [task.id for task in cfg.blind_test_tasks] == ["blind_inventory_badges", "blind_inventory_vouchers"]


def test_explicit_ops_benchmark_config() -> None:
    cfg = load_benchmark_config("configs/benchmark_explicit_ops.yaml")
    assert cfg.name == "benchmark_explicit_ops"
    assert cfg.evolve_iterations == 3
    assert cfg.critic_mode == "off"
    assert [task.id for task in cfg.train_tasks] == ["train_explicit_labels"]
    assert [task.id for task in cfg.dev_probe_tasks] == ["dev_explicit_tags_inline", "dev_explicit_stickers_table"]
    assert [task.id for task in cfg.blind_test_tasks] == [
        "blind_explicit_badges_jsonish",
        "blind_explicit_vouchers_semicolon",
    ]
    assert "+18*52" in cfg.blind_test_tasks[0].instruction
    assert "handed out" not in cfg.blind_test_tasks[0].instruction


def test_explicit_ops_v2_benchmark_config_covers_schema_variants() -> None:
    cfg = load_benchmark_config("configs/benchmark_explicit_ops_v2.yaml")
    assert cfg.name == "benchmark_explicit_ops_v2"
    assert cfg.evolve_iterations == 3
    assert cfg.critic_mode == "off"
    assert cfg.transfer_context_mode == "feedback_only"
    assert cfg.repair_mode == "focused"
    assert [task.id for task in cfg.train_tasks] == [
        "train_explicit_labels_block",
        "train_explicit_cards_jsonish",
    ]
    assert [task.id for task in cfg.dev_probe_tasks] == [
        "dev_explicit_tags_inline",
        "dev_explicit_stickers_table",
        "dev_explicit_vouchers_semicolon",
    ]
    assert [task.id for task in cfg.blind_test_tasks] == [
        "blind_explicit_badges_reordered_jsonish",
        "blind_explicit_tickets_semicolon",
    ]
    all_text = "\n".join(task.instruction for task in cfg.train_tasks + cfg.dev_probe_tasks + cfg.blind_test_tasks)
    assert '"updates"' in all_text
    assert "updates=[" in all_text
    assert "| sign | value |" in all_text
    assert "operations: +" in all_text
    assert "; -10,004;" in all_text
    assert "unit: tickets; initial:" in all_text
    assert "handed out" not in all_text
    assert "redeemed" not in all_text
    assert "voided" not in all_text


def test_unit_conversion_benchmark_config() -> None:
    cfg = load_benchmark_config("configs/benchmark_unit_conversion.yaml")
    assert cfg.name == "benchmark_unit_conversion"
    assert cfg.evolve_iterations == 3
    assert [task.id for task in cfg.train_tasks] == ["train_solution_liters"]
    assert [task.id for task in cfg.dev_probe_tasks] == ["dev_package_grams", "dev_cable_meters"]
    assert [task.id for task in cfg.blind_test_tasks] == ["blind_syrup_milliliters", "blind_rope_centimeters"]


def test_unit_conversion_focused_benchmark_config_exercises_transfer_repair() -> None:
    cfg = load_benchmark_config("configs/benchmark_unit_conversion_focused.yaml")
    assert cfg.name == "benchmark_unit_conversion_focused"
    assert cfg.evolve_iterations == 3
    assert cfg.critic_mode == "off"
    assert cfg.transfer_context_mode == "feedback_only"
    assert cfg.repair_mode == "focused"
    assert cfg.inner_repair_attempts == 1
    assert cfg.teacher_policy_audit is True
    assert cfg.policy_generalization_audit is True
    assert [task.id for task in cfg.train_tasks] == ["train_solution_liters"]
    assert [task.id for task in cfg.dev_probe_tasks] == [
        "dev_package_grams",
        "dev_cable_meters",
        "dev_water_milliliters",
        "dev_syrup_milliliters_poured_out",
    ]
    assert [task.id for task in cfg.blind_test_tasks] == ["blind_syrup_milliliters", "blind_rope_centimeters"]
    all_text = "\n".join(task.instruction for task in cfg.train_tasks + cfg.dev_probe_tasks + cfg.blind_test_tasks)
    assert "liters" in all_text
    assert "milliliters" in all_text
    assert "kilograms" in all_text
    assert "grams" in all_text
    assert "centimeters" in all_text
    assert "meters" in all_text
    assert "poured out" in all_text
    assert "poured out" not in cfg.train_tasks[0].instruction


def test_unit_conversion_focused_blind_rubric_marks_poured_out_subtractive() -> None:
    cfg = load_benchmark_config("configs/benchmark_unit_conversion_focused.yaml")
    blind = next(task for task in cfg.blind_test_tasks if task.id == "blind_syrup_milliliters")
    assert blind.rubric is not None
    assert "poured out" in blind.rubric
    assert "subtractive cue" in blind.rubric


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


def test_impact_report_includes_tool_results(tmp_path: Path) -> None:
    rows = build_impact_report(
        baseline={
            "task": {
                "weak_answer": "0",
                "tool_call": {"name": "signed_sum", "input": {}},
                "tool_result": {"ok": False},
                "runtime_policy_results": [{"requires_tool": True}],
            }
        },
        after={
            "task": {
                "weak_answer": "1",
                "tool_call": {"name": "signed_sum", "input": {"start": 1}},
                "tool_result": {"ok": True, "result": 1},
                "runtime_policy_results": [],
            }
        },
        tasks=[TaskConfig(id="task", instruction="return 1", expected_answer="1")],
        output_path=tmp_path / "impact.json",
    )

    assert rows[0]["before_tool_result"] == {"ok": False}
    assert rows[0]["after_tool_result"] == {"ok": True, "result": 1}
    assert rows[0]["before_runtime_policy_results"] == [{"requires_tool": True}]


def test_build_benchmark_metrics_counts_patch_quality_and_transfer() -> None:
    metrics = build_benchmark_metrics(
        train_summary=[
            {
                "patch_status": "accepted",
                "phase_kind": "full_train",
                "created_at": "2026-01-01T00:00:00+00:00",
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
                "phase_kind": "focused_repair",
                "created_at": "2026-01-01T00:00:10+00:00",
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
    assert metrics["repair_efficiency"]["patch_attempts"] == 2
    assert metrics["repair_efficiency"]["accepted_rate"] == 0.5
    assert metrics["repair_efficiency"]["repair_success"] is True
    assert metrics["repair_efficiency"]["repair_success_via"] == "outer_patch"
    assert metrics["repair_efficiency"]["avg_paths_per_patch_attempt"] == 2.5
    assert metrics["repair_efficiency"]["dev"]["improved"] == 1
    assert metrics["repair_efficiency"]["cost_proxies"]["weak_call_proxy"] == 1
    assert metrics["repair_efficiency"]["cost_proxies"]["train_created_at_span_seconds"] == 10.0


def test_build_benchmark_metrics_separates_blind_transfer() -> None:
    metrics = build_benchmark_metrics(
        train_summary=[],
        impact_rows=[{"before_success": False, "after_success": True, "improved": True, "regressed": False}],
        harness_files_after=[],
        blind_impact_rows=[{"before_success": False, "after_success": False, "improved": False, "regressed": False}],
    )

    assert metrics["dev_transfer"]["improved"] == 1
    assert metrics["blind_transfer"]["improved"] == 0


def test_build_benchmark_metrics_counts_scoped_inner_repair_efficiency() -> None:
    metrics = build_benchmark_metrics(
        train_summary=[
            {
                "patch_status": "rejected",
                "rejected_patch_paths": ["/repo/harness/runtime_policies/force_inventory.py"],
                "contract_validation": [{"ok": False}],
                "inner_repair_attempts": [
                    {
                        "patch_status": "rejected",
                        "focused_repair": True,
                        "rejection_reason": "inner repair patch targets outside allowed repair scope",
                        "context_repair_scope": {
                            "allowed_repair_paths": [
                                "harness/runtime_policies/force_inventory.py",
                                "harness/tests/force_inventory.json",
                            ]
                        },
                    },
                    {
                        "patch_status": "accepted",
                        "focused_repair": True,
                        "context_repair_scope": {
                            "allowed_repair_paths": [
                                "harness/runtime_policies/force_inventory.py",
                                "harness/tests/force_inventory.json",
                            ]
                        },
                    },
                ],
            }
        ],
        impact_rows=[{"before_success": False, "after_success": False, "improved": False, "regressed": False}],
        harness_files_after=[],
        blind_impact_rows=[{"before_success": True, "after_success": False, "improved": False, "regressed": True}],
    )

    repair = metrics["repair_efficiency"]
    assert repair["inner_repair_attempts"] == 2
    assert repair["repair_success"] is True
    assert repair["repair_success_via"] == "scoped_inner_repair"
    assert repair["inner_repair_accepted"] == 1
    assert repair["scoped_inner_repair_attempts"] == 2
    assert repair["scoped_inner_repair_accepted"] == 1
    assert repair["scoped_inner_repair_success"] is True
    assert repair["out_of_scope_rejections"] == 1
    assert repair["blind"]["regressed"] == 1
    assert repair["cost_proxies"]["teacher_call_proxy"] == 3
    assert repair["cost_proxies"]["focused_repair_weak_calls_skipped"] == 2


def test_repair_efficiency_report_aggregates_runs(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "metrics.json").write_text(
        json.dumps(
            {
                "repair_efficiency": {
                    "patch_attempts": 2,
                    "accepted": 1,
                    "rejected": 1,
                    "repair_success": True,
                    "inner_repair_attempts": 1,
                    "inner_repair_accepted": 1,
                    "scoped_inner_repair_attempts": 1,
                    "scoped_inner_repair_accepted": 1,
                    "scoped_inner_repair_success": True,
                    "out_of_scope_rejections": 0,
                    "total_patch_paths": 3,
                    "unique_patch_paths": 2,
                    "cost_proxies": {
                        "teacher_call_proxy": 2,
                        "weak_call_proxy": 1,
                        "focused_repair_weak_calls_skipped": 1,
                    },
                    "dev": {"improved": 1, "regressed": 0},
                    "blind": {"improved": 0, "regressed": 0},
                },
                "patches": {"accepted": 1},
                "dev_transfer": {"improved": 1},
                "blind_transfer": {"improved": 0},
            }
        )
    )
    (run_b / "train_summary.json").write_text(
        json.dumps(
            [
                {
                    "patch_status": "accepted",
                    "applied_patch_paths": ["/repo/harness/tools/calc.py", "/repo/harness/tests/calc.json"],
                    "inner_repair_attempts": [],
                }
            ]
        )
    )
    (run_b / "dev_impact_report.json").write_text(json.dumps([{"improved": False, "regressed": False}]))
    (run_b / "blind_impact_report.json").write_text(json.dumps([{"improved": True, "regressed": False}]))
    (run_b / "harness_files_after.json").write_text(json.dumps(["harness/tools/calc.py"]))

    report = build_repair_efficiency_report([run_a, run_b])

    assert len(report["runs"]) == 2
    assert report["aggregate"]["patch_attempts"] == 3
    assert report["aggregate"]["accepted"] == 2
    assert report["aggregate"]["accepted_rate"] == 0.6667
    assert report["aggregate"]["repair_successes"] == 2
    assert report["aggregate"]["inner_repair_accepted"] == 1
    assert report["aggregate"]["scoped_inner_repair_attempts"] == 1
    assert report["aggregate"]["scoped_inner_repair_accepted"] == 1
    assert report["aggregate"]["scoped_inner_repair_successes"] == 1
    assert report["aggregate"]["dev_improved"] == 1
    assert report["aggregate"]["blind_improved"] == 1
    assert report["aggregate"]["teacher_call_proxy"] == 3
    assert report["aggregate"]["focused_repair_weak_calls_skipped"] == 1


def test_repair_efficiency_report_recovers_interrupted_run_from_phase_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    phase_dir = run_dir / "evolve_train_iter_02"
    phase_dir.mkdir(parents=True)
    (phase_dir / "train.inner_repair_1.json").write_text(json.dumps({"patch_status": "rejected"}))
    (phase_dir / "train.json").write_text(
        json.dumps(
            {
                "task_id": "train",
                "created_at": "2026-01-01T00:00:00+00:00",
                "patch_status": "rejected",
                "rejected_patch_paths": ["/repo/harness/runtime_policies/force_train.py"],
                "contract_validation": [{"ok": False}],
                "inner_repair_attempts": [
                    {
                        "patch_status": "rejected",
                        "focused_repair": True,
                        "context_repair_scope": {
                            "allowed_repair_paths": [
                                "harness/runtime_policies/force_train.py",
                                "harness/tests/force_train.json",
                            ]
                        },
                    }
                ],
            }
        )
    )

    report = build_repair_efficiency_report([run_dir])

    repair = report["runs"][0]["repair_efficiency"]
    assert repair["patch_attempts"] == 1
    assert repair["rejected"] == 1
    assert repair["repair_success"] is False
    assert repair["repair_success_via"] == "none"
    assert repair["inner_repair_attempts"] == 1
    assert repair["scoped_inner_repair_attempts"] == 1
    assert repair["scoped_inner_repair_success"] is False
    assert repair["cost_proxies"]["teacher_call_proxy"] == 2


def test_repair_fixture_deterministically_exposes_scoped_inner_repair(tmp_path: Path) -> None:
    report = run_repair_fixture(tmp_path / "fixture")
    runs = {Path(row["run_dir"]).name: row["repair_efficiency"] for row in report["runs"]}

    assert runs["fixture_full_train"]["patch_attempts"] == 1
    assert runs["fixture_full_train"]["rejected"] == 1
    assert runs["fixture_full_train"]["repair_success"] is False
    assert runs["fixture_full_train"]["repair_success_via"] == "none"
    assert runs["fixture_full_train"]["inner_repair_attempts"] == 0
    assert runs["fixture_full_train"]["scoped_inner_repair_success"] is False
    assert runs["fixture_focused_only"]["rejected"] == 1
    assert runs["fixture_focused_only"]["repair_success"] is False
    assert runs["fixture_focused_only"]["repair_success_via"] == "none"
    assert runs["fixture_focused_only"]["inner_repair_attempts"] == 0
    assert runs["fixture_focused_only"]["scoped_inner_repair_success"] is False
    assert runs["fixture_scoped_inner"]["rejected"] == 1
    assert runs["fixture_scoped_inner"]["repair_success"] is True
    assert runs["fixture_scoped_inner"]["repair_success_via"] == "scoped_inner_repair"
    assert runs["fixture_scoped_inner"]["inner_repair_attempts"] == 1
    assert runs["fixture_scoped_inner"]["inner_repair_accepted"] == 1
    assert runs["fixture_scoped_inner"]["scoped_inner_repair_attempts"] == 1
    assert runs["fixture_scoped_inner"]["scoped_inner_repair_accepted"] == 1
    assert runs["fixture_scoped_inner"]["scoped_inner_repair_success"] is True
    assert report["aggregate"]["rejected"] == 3
    assert report["aggregate"]["repair_successes"] == 1
    assert report["aggregate"]["scoped_inner_repair_attempts"] == 1
    assert report["aggregate"]["scoped_inner_repair_accepted"] == 1
    assert report["aggregate"]["scoped_inner_repair_successes"] == 1


def test_repair_probe_passes_patch_feedback_and_scope_to_teacher(tmp_path: Path) -> None:
    class TeacherClient:
        def __init__(self) -> None:
            self.messages = None

        async def complete(self, messages, temperature=0.2):
            self.messages = messages
            return json.dumps(
                {
                    "diagnosis": "Repair the scoped harness.",
                    "failure_categories": ["runtime_policy"],
                    "harness_patch": "repair the fixture",
                    "patch_type": "runtime_policy",
                    "regression_test": "keep repair scoped",
                    "patch_bundles": [
                        {
                            "target_path": "harness/runtime_policies/force_fixture.py",
                            "action": "create_or_replace",
                            "content": "def evaluate(input: dict) -> dict:\n    return {\"requires_tool\": False}\n",
                        },
                        {
                            "target_path": "harness/tests/force_fixture.json",
                            "action": "create_or_replace",
                            "content": json.dumps(
                                {
                                    "policy": "force_fixture",
                                    "cases": [
                                        {
                                            "input": {
                                                "task_instruction": "Use the signed updates: start=100, updates=[+25, -7]. Return the final count.",
                                                "available_tools": [],
                                                "expected_answer": "118",
                                            },
                                            "expected": {"requires_tool": False},
                                        }
                                    ],
                                }
                            ),
                        },
                    ],
                    "harness_manifest": {
                        "bundle_id": "repair_fixture",
                        "intent": "repair fixture",
                        "allowed_paths": [
                            "harness/runtime_policies/force_fixture.py",
                            "harness/tests/force_fixture.json",
                        ],
                        "artifacts": [
                            {
                                "path": "harness/runtime_policies/force_fixture.py",
                                "type": "runtime_policy",
                                "purpose": "fixture repair",
                            },
                            {
                                "path": "harness/tests/force_fixture.json",
                                "type": "test",
                                "purpose": "fixture repair",
                            },
                        ],
                        "contracts": ["fixture contracts pass"],
                    },
                }
                )

    probe_repo = tmp_path / "repo"
    (probe_repo / "prompts").mkdir(parents=True)
    (probe_repo / "prompts" / "weak_system.md").write_text("weak")
    (probe_repo / "prompts" / "teacher_diagnosis.md").write_text("teacher")
    for subdir in ["guidelines", "skills", "validators", "tools", "runtime_policies", "tests"]:
        (probe_repo / "harness" / subdir).mkdir(parents=True, exist_ok=True)

    teacher = TeacherClient()
    report = asyncio.run(run_repair_probe(tmp_path / "probe", None, teacher=teacher, repo_root=probe_repo))

    assert teacher.messages is not None
    payload = json.loads(teacher.messages[1]["content"])
    assert payload["benchmark_context"]["patch_feedback"]["has_rejections"] is True
    assert "harness/runtime_policies/force_fixture.py" in payload["benchmark_context"]["repair_scope"]["allowed_repair_paths"]
    assert "harness/tests/force_fixture.json" in payload["benchmark_context"]["repair_scope"]["allowed_repair_paths"]
    assert report["repair_success"] is True
    assert report["repair_success_via"] == "scoped_inner_repair"


def test_teacher_prompt_enrichment_lifts_repair_plan_and_boundaries() -> None:
    context = {
        "repair_mode": "focused",
        "repair_scope": {
            "allowed_repair_paths": [
                "harness/runtime_policies/force_fixture.py",
                "harness/tests/force_fixture.json",
            ],
            "failure_kinds": ["runtime_policy"],
            "source_rejected_paths": ["harness/runtime_policies/force_fixture.py"],
            "scope_reason": "runtime policy contract failures should repair only the policy and matching tests",
        },
        "transfer_feedback": {
            "has_transfer_failures": True,
            "failed_tasks": [
                {
                    "task_id": "dev_fixture",
                    "repair_plan": {
                        "primary_axis": "runtime_policy",
                        "allowed_artifact_types": ["runtime_policy", "test"],
                        "required_regression_test": "runtime policy test covering the failed routing case and tool_input",
                    },
                }
            ],
        },
    }

    enriched = enrich_benchmark_context(context)

    assert enriched is not None
    assert enriched["repair_plan"]["primary_axis"] == "runtime_policy"
    assert enriched["artifact_boundaries"]["allowed_artifact_types"] == ["runtime_policy", "test"]
    assert "allowed_repair_paths" in enriched["artifact_boundaries"]
    assert enriched["artifact_boundaries"]["required_regression_test"] == "runtime policy test covering the failed routing case and tool_input"


def test_teacher_prompt_mentions_raw_patch_bundle_content() -> None:
    text = Path("prompts/teacher_diagnosis.md").read_text()
    assert "patch_bundles.content" in text
    assert "actual file text" in text
    assert "double-escaped" in text


def test_teacher_payload_preserves_core_fields() -> None:
    payload = build_teacher_payload(
        task_id="task",
        task_instruction="repair",
        expected_answer="1",
        rubric="rubric",
        weak_system_prompt="weak system",
        weak_answer="answer",
        initial_weak_answer="initial",
        tool_call={"name": "adder", "input": {"a": 1}},
        tool_result={"ok": True, "total": 1},
        runtime_policy_results=[{"requires_tool": True}],
        benchmark_context=None,
    )

    assert payload["task_id"] == "task"
    assert payload["task_instruction"] == "repair"
    assert payload["weak_system_prompt"] == "weak system"
    assert payload["tool_call"] == {"name": "adder", "input": {"a": 1}}
    assert payload["runtime_policy_results"] == [{"requires_tool": True}]
    assert "benchmark_context" not in payload


def test_repair_family_cases_cover_distinct_mechanisms() -> None:
    cases = build_repair_family_cases()

    assert [case.case_id for case in cases] == ["tool_policy_pair", "tool_contract_repair", "fallback_rejected_paths"]
    assert [case.mechanism for case in cases] == ["tool_policy_pair", "tool", "prompt_guideline"]
    assert cases[0].manifest is not None
    assert cases[1].manifest is not None
    assert cases[1].manifest.bundle_id == "signed_sum_tool"
    assert cases[2].manifest is None
    assert cases[2].bad_policy_bundles[0].action == "append"
    assert cases[0].dev_probe_tasks is not None
    assert [task.id for task in cases[0].dev_probe_tasks] == ["dev_fixture_signed_updates"]
    assert cases[0].blind_test_tasks is not None
    assert [task.id for task in cases[0].blind_test_tasks] == ["blind_fixture_signed_updates"]
    assert cases[1].dev_probe_tasks is not None
    assert [task.id for task in cases[1].dev_probe_tasks] == ["dev_signed_sum_tool_updates"]
    assert cases[1].blind_test_tasks is not None
    assert [task.id for task in cases[1].blind_test_tasks] == ["blind_signed_sum_tool_updates"]
    assert cases[2].mechanism_only is True


def test_repair_family_can_include_diagnostic_scope_variant() -> None:
    cases = build_repair_family_cases(include_diagnostics=True)

    assert [case.case_id for case in cases][-1] == "tool_policy_pair_scoped"
    assert cases[-1].diagnostic is True
    assert cases[-1].repair_scope_override is not None
    assert cases[-1].repair_scope_override["allowed_repair_paths"] == [
        "harness/runtime_policies/force_fixture.py",
        "harness/tests/force_fixture.json",
    ]


def test_repair_family_tool_case_uses_manifest() -> None:
    cases = build_repair_family_cases()

    tool_case = cases[1]
    assert tool_case.manifest is not None
    assert tool_case.manifest.bundle_id == "signed_sum_tool"
    assert "harness/tools/signed_sum.py" in tool_case.manifest.allowed_paths
    assert "harness/tests/signed_sum.json" in tool_case.manifest.allowed_paths


def test_repair_family_weak_system_loads_harness_tools(tmp_path: Path) -> None:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "weak_system.md").write_text("base weak prompt", encoding="utf-8")
    for subdir in ["guidelines", "skills", "validators", "tools"]:
        (tmp_path / "harness" / subdir).mkdir(parents=True)
    (tmp_path / "harness" / "tools" / "signed_sum.py").write_text(
        """
def run(input: dict) -> dict:
    return {"ok": True, "result": 1}
""".strip(),
        encoding="utf-8",
    )

    system_prompt = _weak_system(tmp_path)

    assert "base weak prompt" in system_prompt
    assert "Harness tool specs" in system_prompt
    assert "Tool module: signed_sum" in system_prompt


def test_probe_filter_cases_use_stricter_candidates() -> None:
    cases = build_probe_filter_cases()

    tool_case = next(case for case in cases if case.case_id == "tool_contract_repair")
    assert tool_case.dev_probe_tasks is not None
    assert len(tool_case.dev_probe_tasks) >= 2
    assert tool_case.dev_probe_tasks[0].expected_answer == "7208"
    assert tool_case.blind_test_tasks is not None
    assert len(tool_case.blind_test_tasks) >= 2
    assert tool_case.blind_test_tasks[0].expected_answer == "39534"


def test_probe_filter_report_records_baseline_failures(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "prompts").mkdir(parents=True)
    (repo_root / "prompts" / "weak_system.md").write_text("weak", encoding="utf-8")
    (repo_root / "prompts" / "teacher_diagnosis.md").write_text("teacher", encoding="utf-8")
    for subdir in ["guidelines", "skills", "validators", "tools", "runtime_policies", "tests"]:
        (repo_root / "harness" / subdir).mkdir(parents=True, exist_ok=True)
    (repo_root / "harness" / "tools" / "signed_sum.py").write_text(
        """
def run(input: dict) -> dict:
    total = int(input.get("start", 0))
    for value in input.get("updates", []):
        total += int(str(value).replace(",", ""))
    return {"ok": True, "result": total}
""".strip(),
        encoding="utf-8",
    )

    class WeakClient:
        async def complete(self, messages, temperature=0.2):
            return "I do not know"

    class TeacherClient:
        async def complete(self, messages, temperature=0.2):
            return """{\n  \"diagnosis\": \"ok\",\n  \"failure_categories\": [],\n  \"harness_patch\": \"\",\n  \"patch_type\": \"prompt_guideline\",\n  \"regression_test\": \"\",\n  \"patch_bundles\": [],\n  \"confidence\": 0.0\n}"""

    report = asyncio.run(
        run_probe_filter(
            tmp_path / "probe_filter",
            None,
            repo_root=repo_root,
            weak=WeakClient(),
            teacher=TeacherClient(),
        )
    )

    assert report["summary"]["cases"] >= 1
    assert "probe_filter_report.json" in {p.name for p in (tmp_path / "probe_filter").iterdir()}


def test_transfer_tight_mode_uses_probe_filter_cases() -> None:
    cases = build_probe_filter_cases()
    assert [task.id for task in cases[0].dev_probe_tasks or []][0].startswith("dev_filter_")
    assert [task.id for task in cases[1].dev_probe_tasks or []][0].startswith("dev_filter_")
