from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentdistill.tools import ToolRegistry


def validate_tool_tests(repo_root: Path, tool_name: str) -> dict[str, Any]:
    tests_path = repo_root / "harness" / "tests" / f"{tool_name}.json"
    if not tests_path.exists():
        return {"ok": True, "reason": "no tool tests found"}

    try:
        data = json.loads(tests_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "reason": f"invalid JSON test file: {exc}"}

    if not isinstance(data, dict) or data.get("tool") != tool_name:
        return {"ok": False, "reason": "test file must be an object with matching tool name"}

    cases = data.get("cases", [])
    if not isinstance(cases, list) or not cases:
        return {"ok": False, "reason": "test file must contain a non-empty cases list"}

    tools = ToolRegistry(repo_root / "harness" / "tools")
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
        actual = tools.call(tool_name, payload)
        ok, mismatch_reason = _matches_expected(expected, actual)
        if not ok:
            failures.append(
                {
                    "case_index": idx,
                    "reason": mismatch_reason,
                    "input": payload,
                    "expected": expected,
                    "actual": actual,
                }
            )

    if failures:
        return {"ok": False, "reason": "one or more tool tests failed", "failures": failures}
    return {"ok": True, "reason": "all tool tests passed", "num_cases": len(cases)}


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
