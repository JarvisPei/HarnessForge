from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agentdistill.config import TaskConfig
from agentdistill.tool_validation import load_json_test_file, validate_tool_tests
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


def validate_runtime_policy_tests(repo_root: Path, policy_path: Path) -> dict[str, Any]:
    policy_name = policy_path.stem
    tests_path, data, error = load_json_test_file(repo_root, policy_name)
    if error is not None:
        return {**error, "policy": policy_name}
    if not isinstance(data, dict) or data.get("policy") != policy_name:
        return {"ok": False, "reason": "test file must be an object with matching policy name", "policy": policy_name}
    result = validate_runtime_policy_case_data(repo_root, policy_path, data)
    if result.get("ok") is True:
        result["tests_path"] = str(tests_path)
    return result


def validate_runtime_policy_case_data(
    repo_root: Path,
    policy_path: Path,
    data: dict[str, Any],
    reason: str = "all policy tests passed",
) -> dict[str, Any]:
    policy_name = policy_path.stem
    if data.get("policy") != policy_name:
        return {"ok": False, "reason": "test file must be an object with matching policy name", "policy": policy_name}
    cases = data.get("cases", [])
    if not isinstance(cases, list) or not cases:
        return {"ok": False, "reason": "policy test file must contain a non-empty cases list", "policy": policy_name}

    tools = ToolRegistry(repo_root / "harness" / "tools")
    policies = RuntimePolicyRegistry(policy_path.parent)
    failures = []
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            failures.append({"case_index": idx, "reason": "case must be an object"})
            continue
        payload = case.get("input", {})
        expected = case.get("expected", {})
        if not isinstance(payload, dict) or not isinstance(expected, dict):
            failures.append({"case_index": idx, "reason": "case must have input and expected objects"})
            continue
        policy_payload = {
            "task_instruction": payload.get("task_instruction", ""),
            "initial_answer": payload.get("initial_answer", ""),
            "tool_call": payload.get("tool_call"),
            "available_tools": payload.get("available_tools", tools.names),
            "expected_answer": payload.get("expected_answer"),
            "rubric": payload.get("rubric"),
        }
        results = [result for result in policies.evaluate(policy_payload) if result.get("policy") == policy_name]
        actual = results[0] if results else {"policy": policy_name, "requires_tool": False}
        ok, mismatch_reason = _matches_expected(expected, actual)
        tool_result = None
        expected_tool_result = case.get("expected_tool_result")
        if ok and expected_tool_result is not None:
            if not isinstance(expected_tool_result, dict):
                ok = False
                mismatch_reason = "expected_tool_result must be an object"
            elif actual.get("requires_tool") is not True or not isinstance(actual.get("tool_name"), str) or not isinstance(actual.get("tool_input"), dict):
                ok = False
                mismatch_reason = "policy did not produce a valid forced tool call"
            else:
                tool_result = tools.call(actual["tool_name"], actual["tool_input"])
                ok, mismatch_reason = _matches_expected(expected_tool_result, tool_result)
        if not ok:
            failures.append(
                {
                    "case_index": idx,
                    "reason": mismatch_reason,
                    "input": payload,
                    "expected": expected,
                    "actual": actual,
                    "tool_result": tool_result,
                }
            )

    if failures:
        return {"ok": False, "reason": "one or more policy tests failed", "policy": policy_name, "failures": failures}
    return {"ok": True, "reason": reason, "policy": policy_name, "num_cases": len(cases)}


def validate_runtime_policy_generalization(repo_root: Path, policy_path: Path) -> dict[str, Any]:
    policy_name = policy_path.stem
    tests_path, data, error = load_json_test_file(repo_root, policy_name)
    if error is not None:
        return {**error, "policy": policy_name}
    if not isinstance(data, dict) or data.get("policy") != policy_name:
        return {"ok": False, "reason": "test file must be an object with matching policy name", "policy": policy_name}

    tools = ToolRegistry(repo_root / "harness" / "tools")
    policies = RuntimePolicyRegistry(policy_path.parent)
    failures = []
    audited = 0
    for idx, case in enumerate(data.get("cases", [])):
        if not isinstance(case, dict):
            continue
        payload = case.get("input", {})
        expected = case.get("expected", {})
        expected_tool_result = case.get("expected_tool_result")
        if not isinstance(payload, dict) or not isinstance(expected, dict):
            continue
        if expected.get("requires_tool") is not True or not isinstance(expected.get("tool_name"), str):
            continue
        instruction = payload.get("task_instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        for mutation in _policy_generalization_mutations(payload, expected_tool_result if isinstance(expected_tool_result, dict) else None):
            audited += 1
            mutated_payload = {
                "task_instruction": mutation["task_instruction"],
                "initial_answer": payload.get("initial_answer", ""),
                "tool_call": payload.get("tool_call"),
                "available_tools": payload.get("available_tools", tools.names),
                "expected_answer": mutation.get("expected_answer", payload.get("expected_answer")),
                "rubric": payload.get("rubric"),
            }
            results = [result for result in policies.evaluate(mutated_payload) if result.get("policy") == policy_name]
            actual = results[0] if results else {"policy": policy_name, "requires_tool": False}
            if actual.get("requires_tool") is not True or actual.get("tool_name") != expected.get("tool_name"):
                failures.append(
                    {
                        "case_index": idx,
                        "mutation": mutation["mutation"],
                        "reason": "policy trigger is not invariant to schema-preserving wording changes",
                        "mutated_instruction": mutation["task_instruction"],
                        "expected": {"requires_tool": True, "tool_name": expected.get("tool_name")},
                        "actual": actual,
                    }
                )
                continue
            if not isinstance(actual.get("tool_input"), dict):
                failures.append(
                    {
                        "case_index": idx,
                        "mutation": mutation["mutation"],
                        "reason": "policy produced no object tool_input after schema-preserving wording changes",
                        "mutated_instruction": mutation["task_instruction"],
                        "actual": actual,
                    }
                )
                continue
            tool_result = tools.call(actual["tool_name"], actual["tool_input"])
            expected_answer = payload.get("expected_answer")
            if tool_result.get("ok") is not True or (
                isinstance(expected_answer, str) and not _tool_result_matches_expected(expected_answer, tool_result)
            ):
                failures.append(
                    {
                        "case_index": idx,
                        "mutation": mutation["mutation"],
                        "reason": "forced tool result failed after schema-preserving wording changes",
                        "mutated_instruction": mutation["task_instruction"],
                        "actual": actual,
                        "tool_result": tool_result,
                    }
                )

    if failures:
        return {
            "ok": False,
            "reason": "one or more policy generalization audits failed",
            "policy": policy_name,
            "failures": failures,
        }
    return {
        "ok": True,
        "reason": "policy generalization audit passed" if audited else "no eligible policy generalization audit cases",
        "policy": policy_name,
        "tests_path": str(tests_path),
        "num_cases": audited,
    }


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


MEASUREMENT_UNITS = {
    "cm",
    "centimeter",
    "centimeters",
    "g",
    "gram",
    "grams",
    "kg",
    "kilogram",
    "kilograms",
    "l",
    "liter",
    "liters",
    "metre",
    "metres",
    "meter",
    "meters",
    "milliliter",
    "milliliters",
    "millilitre",
    "millilitres",
    "ml",
}


def _policy_generalization_mutations(payload: dict[str, Any], expected_tool_result: dict[str, Any] | None) -> list[dict[str, str]]:
    instruction = payload.get("task_instruction")
    if not isinstance(instruction, str):
        return []

    mutations: list[dict[str, str]] = []
    surface_term = _surface_count_noun(payload, expected_tool_result)
    if surface_term is not None:
        mutated_instruction = _replace_surface_term(instruction, surface_term)
        if mutated_instruction != instruction:
            mutation = {
                "mutation": "surface_entity_rename",
                "task_instruction": mutated_instruction,
            }
            if payload.get("expected_answer") is not None:
                mutation["expected_answer"] = _replace_surface_term(str(payload.get("expected_answer", "")), surface_term)
            mutations.append(mutation)

    for mutation_name, mutated_instruction in _schema_preserving_paraphrases(instruction):
        if mutated_instruction != instruction:
            mutations.append({"mutation": mutation_name, "task_instruction": mutated_instruction})

    for mutation in _signed_operation_format_mutations(instruction, payload):
        mutations.append(mutation)

    deduped = []
    seen = set()
    for mutation in mutations:
        key = (mutation["mutation"], mutation["task_instruction"])
        if key not in seen:
            seen.add(key)
            deduped.append(mutation)
    return deduped


def _signed_operation_format_mutations(instruction: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    parsed = _extract_signed_operation_schema(instruction)
    if parsed is None:
        return []
    unit = parsed["unit"]
    start = parsed["start"]
    updates = parsed["updates"]
    expected_answer = payload.get("expected_answer")
    mutations = [
        {
            "mutation": "signed_ops_semicolon_format",
            "task_instruction": (
                f"Compute the final count from the explicit signed operations: "
                f"unit: {unit}; initial: {start}; operations: {'; '.join(updates)}. "
                f"Return the final count in {unit} with one short explanation sentence."
            ),
        },
        {
            "mutation": "signed_ops_reordered_jsonish_format",
            "task_instruction": (
                "Determine the final inventory count.\n\n"
                + json.dumps({"updates": updates, "initial": start, "unit": unit}, indent=2)
                + f"\n\nReturn the final count in {unit} with one short explanation sentence."
            ),
        },
    ]
    if isinstance(expected_answer, str):
        for mutation in mutations:
            mutation["expected_answer"] = expected_answer
    return mutations


def _extract_signed_operation_schema(instruction: str) -> dict[str, Any] | None:
    start_match = re.search(r"\b(?:start|initial)\s*[:=]\s*([+-]?\d[\d,]*)", instruction, flags=re.I)
    if not start_match:
        return None
    unit = _extract_unit(instruction)
    if unit is None:
        unit = "items"
    updates = _extract_signed_operations(instruction)
    if len(updates) < 2:
        return None
    return {"unit": unit, "start": start_match.group(1), "updates": updates}


def _extract_unit(instruction: str) -> str | None:
    patterns = [
        r"\bunit\s*[:=]\s*([A-Za-z][A-Za-z-]*)",
        r"\bfinal\s+count\s+in\s+([A-Za-z][A-Za-z-]*)\b",
        r"\b([A-Za-z][A-Za-z-]*)\s+remain\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, instruction, flags=re.I)
        if match:
            return match.group(1)
    return None


def _extract_signed_operations(instruction: str) -> list[str]:
    updates_match = re.search(r"\b(?:updates|operations)\s*[:=]\s*\[([^\]]+)\]", instruction, flags=re.I | re.S)
    if updates_match:
        return [item.strip().strip("\"'") for item in updates_match.group(1).split(",") if item.strip()]

    table_ops: list[str] = []
    for sign, value in re.findall(r"^\s*\|\s*(\+|-)\s*\|\s*([^|]+?)\s*\|", instruction, flags=re.M):
        table_ops.append(f"{sign}{value.strip()}")
    if table_ops:
        return table_ops

    line_ops = re.findall(r"^\s*([+-]\s*\d[\d,\s]*(?:\*\s*\d[\d,\s]*)?)\s*$", instruction, flags=re.M)
    if line_ops:
        return [op.replace(" ", "") for op in line_ops]

    operations_match = re.search(r"\boperations\s*:\s*(.+?)(?:\.\s|$)", instruction, flags=re.I | re.S)
    if operations_match:
        return [item.strip() for item in operations_match.group(1).split(";") if item.strip()]

    return []


def _surface_count_noun(payload: dict[str, Any], expected_tool_result: dict[str, Any] | None) -> str | None:
    candidates = []
    if expected_tool_result is not None and isinstance(expected_tool_result.get("unit"), str):
        candidates.append(expected_tool_result["unit"])
    expected_answer = payload.get("expected_answer")
    if isinstance(expected_answer, str):
        match = re.search(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s+([A-Za-z][A-Za-z-]*)\s+(?:remain|left)\b", expected_answer, flags=re.I)
        if match:
            candidates.append(match.group(1))
    instruction = payload.get("task_instruction")
    if isinstance(instruction, str):
        match = re.search(r"\bhow\s+many\s+([A-Za-z][A-Za-z-]*)\s+(?:remain|are\s+left|left)\b", instruction, flags=re.I)
        if match:
            candidates.append(match.group(1))
    for candidate in candidates:
        term = candidate.lower().strip()
        if term and term not in MEASUREMENT_UNITS:
            return term
    return None


def _schema_preserving_paraphrases(text: str) -> list[tuple[str, str]]:
    rewrites = [
        ("initial_cue", r"\bhad\s+(\d)", r"started with \1"),
        ("initial_cue", r"\bbegan\s+with\s+(\d)", r"started with \1"),
        ("product_container", r"\bsheets?\s+with\s+", "crates with "),
        ("product_container", r"\bpacks?\s+with\s+", "bags with "),
        ("direct_subtract_verb", r"\bshipped\s+(\d)", r"gave away \1"),
        ("direct_subtract_verb", r"\bsold\s+(\d)", r"gave away \1"),
        ("direct_subtract_verb", r"\bused\s+(\d)", r"removed \1"),
        ("direct_subtract_verb", r"\blost\s+(\d)", r"removed \1"),
        ("direct_add_verb", r"\breceived\s+(\d)", r"got \1"),
        ("direct_add_verb", r"\bbought\s+(\d)", r"acquired \1"),
    ]
    variants = []
    for name, pattern, replacement in rewrites:
        mutated, count = re.subn(pattern, replacement, text, count=1, flags=re.I)
        if count:
            variants.append((name, mutated))
    return variants


def _replace_surface_term(text: str, term: str) -> str:
    singular = term[:-1] if term.endswith("s") and len(term) > 3 else term
    plural = term if term.endswith("s") else f"{term}s"
    replacements = [(plural, "daxels"), (singular, "daxel")]
    result = text
    for source, target in sorted(replacements, key=lambda pair: len(pair[0]), reverse=True):
        result = re.sub(rf"\b{re.escape(source)}\b", target, result, flags=re.I)
    return result


def _matches_expected(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, str]:
    for key, exp_value in expected.items():
        if key not in actual:
            return False, f"missing key: {key}"
        if not _subset_match(exp_value, actual[key]):
            return False, f"value mismatch for key: {key}"
    return True, "ok"


def _subset_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(key in actual and _subset_match(value, actual[key]) for key, value in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) > len(actual):
            return False
        return all(_subset_match(exp_item, actual[idx]) for idx, exp_item in enumerate(expected))
    return expected == actual
