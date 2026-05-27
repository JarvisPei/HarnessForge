from __future__ import annotations

import json
from typing import Any


def build_teacher_messages(
    teacher_system: str,
    *,
    task_id: str,
    task_instruction: str,
    expected_answer: str | None,
    rubric: str | None,
    weak_system_prompt: str,
    weak_answer: str,
    initial_weak_answer: str,
    tool_call: dict[str, object] | None,
    tool_result: dict[str, object] | None,
    runtime_policy_results: list[dict[str, object]],
    benchmark_context: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    payload = build_teacher_payload(
        task_id=task_id,
        task_instruction=task_instruction,
        expected_answer=expected_answer,
        rubric=rubric,
        weak_system_prompt=weak_system_prompt,
        weak_answer=weak_answer,
        initial_weak_answer=initial_weak_answer,
        tool_call=tool_call,
        tool_result=tool_result,
        runtime_policy_results=runtime_policy_results,
        benchmark_context=benchmark_context,
    )
    return [
        {"role": "system", "content": teacher_system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_architect_sketch_messages(
    teacher_system: str,
    *,
    task_id: str,
    task_instruction: str,
    expected_answer: str | None,
    rubric: str | None,
    weak_system_prompt: str,
    weak_answer: str,
    initial_weak_answer: str,
    tool_call: dict[str, object] | None,
    tool_result: dict[str, object] | None,
    runtime_policy_results: list[dict[str, object]],
    benchmark_context: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    payload = build_teacher_payload(
        task_id=task_id,
        task_instruction=task_instruction,
        expected_answer=expected_answer,
        rubric=rubric,
        weak_system_prompt=weak_system_prompt,
        weak_answer=weak_answer,
        initial_weak_answer=initial_weak_answer,
        tool_call=tool_call,
        tool_result=tool_result,
        runtime_policy_results=runtime_policy_results,
        benchmark_context=benchmark_context,
    )
    return [
        {"role": "system", "content": ARCHITECT_SKETCH_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_staged_bundle_messages(
    teacher_system: str,
    *,
    sketch: dict[str, object],
    task_id: str,
    task_instruction: str,
    expected_answer: str | None,
    rubric: str | None,
    weak_system_prompt: str,
    weak_answer: str,
    initial_weak_answer: str,
    tool_call: dict[str, object] | None,
    tool_result: dict[str, object] | None,
    runtime_policy_results: list[dict[str, object]],
    benchmark_context: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    payload = build_teacher_payload(
        task_id=task_id,
        task_instruction=task_instruction,
        expected_answer=expected_answer,
        rubric=rubric,
        weak_system_prompt=weak_system_prompt,
        weak_answer=weak_answer,
        initial_weak_answer=initial_weak_answer,
        tool_call=tool_call,
        tool_result=tool_result,
        runtime_policy_results=runtime_policy_results,
        benchmark_context=benchmark_context,
    )
    payload["architect_sketch"] = sketch
    return [
        {"role": "system", "content": STAGED_BUNDLE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_teacher_payload(
    *,
    task_id: str,
    task_instruction: str,
    expected_answer: str | None,
    rubric: str | None,
    weak_system_prompt: str,
    weak_answer: str,
    initial_weak_answer: str,
    tool_call: dict[str, object] | None,
    tool_result: dict[str, object] | None,
    runtime_policy_results: list[dict[str, object]],
    benchmark_context: dict[str, object] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "task_instruction": task_instruction,
        "expected_answer": expected_answer,
        "rubric": rubric,
        "weak_system_prompt": weak_system_prompt,
        "weak_answer": weak_answer,
        "initial_weak_answer": initial_weak_answer,
        "tool_call": tool_call,
        "tool_result": tool_result,
        "runtime_policy_results": runtime_policy_results,
    }
    enriched_context = enrich_benchmark_context(benchmark_context)
    if enriched_context is not None:
        payload["benchmark_context"] = enriched_context
    return payload


ARCHITECT_SKETCH_SYSTEM_PROMPT = """You are the frontier architect for a weak-model harness.
Your job is to triage the trace and choose the smallest harness-development route.
Return JSON only. Do not write code, markdown patch content, full tests, or patch_bundles.
Choose route:
- text_patch: the likely repair is only guideline, skill, or validator text.
- executable_patch: the likely repair needs a tool, runtime policy, or harness/tests JSON.
- no_patch: no harness change is justified.
- one_pass: unsure; let the normal one-pass patch generator handle it.
For executable_patch, keep the sketch short and bounded:
{
  "route": "executable_patch",
  "patch_type": "tool|runtime_policy",
  "artifacts": [{"path": "harness/...", "type": "tool|runtime_policy|test", "purpose": "..."}],
  "capability": "one sentence",
  "operation_semantics": ["one or two invariants"],
  "test_axes": ["train case", "one schema variation", "one negative/non-trigger case"],
  "complexity_budget": {"max_artifacts": 4, "max_test_cases": 4, "max_code_files": 2},
  "rationale": "one sentence"
}
Artifact path rules:
- tool artifacts must be Python files: harness/tools/<name>.py.
- runtime_policy artifacts must be Python files: harness/runtime_policies/<name>.py.
- tests must be JSON files: harness/tests/<same_tool_or_policy_name>.json.
- every tool or runtime_policy artifact needs a same-stem test artifact.
Activation rule:
- If the weak trace did not already call a tool and the sketch adds a tool, include a runtime_policy artifact that can force that tool plus its same-stem test artifact. A tool without an activation path will not affect future weak-model runs.
For text_patch or one_pass, artifacts may be empty or only text artifacts. Keep rationale to one sentence."""


STAGED_BUNDLE_SYSTEM_PROMPT = """You are the harness developer in a staged harness-distillation workflow.
You receive a weak-model trace and a frozen architect_sketch. Your job is to implement only that sketch as a valid harness patch bundle.

Return JSON only with these fields:
- diagnosis: concise explanation of the weak-model failure and the implemented harness change.
- failure_categories: list using prompt_guideline, skill, tool, validator, state_representation, runtime_policy.
- harness_patch: short description of the files changed.
- patch_type: one of prompt_guideline, skill, tool, validator, state_representation, runtime_policy.
- regression_test: concise future regression test description.
- policy_audit_cases: optional map from runtime policy name to extra policy test cases.
- patch_bundles: list of patch objects.
- harness_manifest: required for any tool, runtime_policy, or test file.
- patch_bundle: the first patch object, kept for backward compatibility.
- confidence: number from 0 to 1.

Patch object schema:
- target_path: one relative path under harness/guidelines, harness/skills, harness/validators, harness/tools, harness/runtime_policies, or harness/tests.
- action: create_or_replace.
- content: complete file text to write directly to disk. Do not JSON-escape newlines inside Python or markdown content.
- rationale: why this change should help future weak-model runs.

Hard constraints:
- Write only target_path values listed in architect_sketch.artifacts.
- Do not add files outside the sketch.
- Do not exceed architect_sketch.complexity_budget.
- If the sketch includes a tool and runtime_policy, implement both so future weak-model runs can actually call the tool.
- For executable sketches, include tests for the listed test_axes and a harness_manifest generalization_contract.
- Do not change expected values to make code pass; preserve operation_semantics.
- If the frozen sketch is insufficient, return a text diagnosis with no patch_bundles explaining the missing scope instead of adding files.

Harness file interface:
- tool files live at harness/tools/<name>.py and expose exactly def run(input: dict) -> dict.
- runtime policy files live at harness/runtime_policies/<name>.py and expose exactly def evaluate(input: dict) -> dict.
- test files live at harness/tests/<name>.json and must share the same stem as the tool or runtime policy they test.
- skill, guideline, and validator files are markdown under harness/skills, harness/guidelines, and harness/validators.

Executable harness rules:
- Python must be deterministic and self-contained. Do not use network, filesystem, subprocess, eval, exec, or non-standard-library imports.
- Tools return JSON-serializable dictionaries and should expose trace fields for operation semantics such as selected rows, per-row contributions, source columns, signs, units, conversion factors, and final formula when relevant.
- Runtime policies should be thin routers when possible. If they normalize or rewrite task text before tool use, include provenance in tool_input so tests can inspect source-to-normalized mappings.

Harness manifest schema for executable bundles:
- bundle_id: short safe identifier.
- intent: why this bundle should improve future weak-model runs.
- allowed_paths: exact target_path values from patch_bundles.
- artifacts: list of objects with path, type, and purpose.
- contracts: list of validation expectations.
- generalization_contract:
  - capability: concise domain-neutral capability.
  - expected_variations: supported surface/schema variations.
  - excluded_variations: intentionally unsupported variations.
  - required_tests: exact harness/tests paths from this manifest.
  - operation_semantics: semantic invariants copied from or consistent with architect_sketch.operation_semantics.
  - semantic_trace_requirements: intermediate fields tests/results expose for audit.

Tool test schema:
{
  "tool": "<name>",
  "cases": [
    {"input": {...}, "expected": {"ok": true}}
  ]
}

Runtime policy test schema:
{
  "policy": "<name>",
  "cases": [
    {
      "input": {"task_instruction": "...", "initial_answer": "", "available_tools": ["<tool>"], "expected_answer": "..."},
      "expected": {"requires_tool": true, "tool_name": "<tool>", "tool_input": {...}},
      "expected_tool_result": {"ok": true}
    }
  ]
}

Generalization discipline:
- Do not hard-code final answers, task IDs, or one-off strings that only solve the observed example.
- Preserve the frozen operation semantics. Do not alter per-row vs group-level scope, sign polarity, unit direction, join keys, or output format just to make a test pass.
- Prefer small helper functions and explicit parsing over clever brittle expressions.
- If the sketch asks for an invalid or insufficient scope, return no patch_bundles and explain the missing scope."""


def enrich_benchmark_context(benchmark_context: dict[str, object] | None) -> dict[str, object] | None:
    if not benchmark_context:
        return None
    context = dict(benchmark_context)
    repair_plan = _derive_repair_plan(context)
    if repair_plan is not None:
        context["repair_plan"] = repair_plan
    artifact_boundaries = _derive_artifact_boundaries(context, repair_plan)
    if artifact_boundaries is not None:
        context["artifact_boundaries"] = artifact_boundaries
    return context


def _derive_repair_plan(context: dict[str, object]) -> dict[str, object] | None:
    repair_plan = context.get("repair_plan")
    if isinstance(repair_plan, dict):
        return dict(repair_plan)

    transfer_feedback = context.get("transfer_feedback")
    if isinstance(transfer_feedback, dict):
        transfer_plan = transfer_feedback.get("repair_plan")
        if isinstance(transfer_plan, dict):
            return dict(transfer_plan)
        failed_tasks = transfer_feedback.get("failed_tasks")
        if isinstance(failed_tasks, list):
            for failed_task in failed_tasks:
                if not isinstance(failed_task, dict):
                    continue
                task_plan = failed_task.get("repair_plan")
                if isinstance(task_plan, dict):
                    return dict(task_plan)

    repair_scope = context.get("repair_scope")
    if isinstance(repair_scope, dict):
        allowed_artifact_types = _allowed_artifact_types_from_scope(repair_scope)
        if allowed_artifact_types:
            return {
                "primary_axis": "artifact_scope",
                "allowed_artifact_types": allowed_artifact_types,
                "required_regression_test": "regression test covering the rejected contract on the allowed repair paths",
            }

    patch_feedback = context.get("patch_feedback")
    if isinstance(patch_feedback, dict) and patch_feedback.get("has_rejections"):
        return {
            "primary_axis": "artifact_scope",
            "allowed_artifact_types": ["tool", "runtime_policy", "test", "guideline", "validator", "skill"],
            "required_regression_test": "regression test covering the rejected contract",
        }

    return None


def _derive_artifact_boundaries(
    context: dict[str, object],
    repair_plan: dict[str, object] | None,
) -> dict[str, object] | None:
    repair_scope = context.get("repair_scope")
    if not isinstance(repair_scope, dict) and repair_plan is None:
        return None

    boundaries: dict[str, object] = {}
    if isinstance(repair_scope, dict):
        for key in ["allowed_repair_paths", "failure_kinds", "source_rejected_paths", "scope_reason"]:
            value = repair_scope.get(key)
            if value is not None:
                boundaries[key] = value
    if isinstance(repair_plan, dict):
        if repair_plan.get("allowed_artifact_types") is not None:
            boundaries["allowed_artifact_types"] = repair_plan["allowed_artifact_types"]
        if repair_plan.get("required_regression_test") is not None:
            boundaries["required_regression_test"] = repair_plan["required_regression_test"]
    return boundaries or None


def _allowed_artifact_types_from_scope(repair_scope: dict[str, object]) -> list[str]:
    allowed_paths = repair_scope.get("allowed_repair_paths", [])
    if not isinstance(allowed_paths, list):
        return []
    artifact_types: list[str] = []
    for path in allowed_paths:
        if not isinstance(path, str):
            continue
        parts = path.split("/")
        if len(parts) != 3 or parts[0] != "harness":
            continue
        artifact_type = parts[1]
        if artifact_type == "tools":
            artifact_types.extend(["tool", "test"])
        elif artifact_type == "runtime_policies":
            artifact_types.extend(["runtime_policy", "test"])
        elif artifact_type == "tests":
            artifact_types.append("test")
        elif artifact_type == "guidelines":
            artifact_types.append("guideline")
        elif artifact_type == "skills":
            artifact_types.append("skill")
        elif artifact_type == "validators":
            artifact_types.append("validator")
    return sorted(set(artifact_types))
