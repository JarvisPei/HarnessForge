from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentdistill.config import TaskConfig
from agentdistill.contracts import validate_runtime_policy_case_data
from agentdistill.diagnosis import PatchBundle
from agentdistill.models import ChatClient


def parse_critic_audit(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"parse_status": "unparsed", "audit_cases": [], "raw": raw}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"parse_status": "unparsed", "audit_cases": [], "raw": raw}
    cases = payload.get("audit_cases", [])
    if not isinstance(cases, list):
        cases = []
    return {
        "parse_status": "parsed",
        "audit_cases": [case for case in cases if isinstance(case, dict)],
        "rationale": payload.get("rationale", ""),
        "raw": raw,
    }


async def request_policy_audit_cases(
    critic: ChatClient,
    critic_system: str,
    task: TaskConfig,
    patch_bundles: list[PatchBundle],
    policy_name: str,
    existing_policy_tests: dict[str, Any] | None,
    max_cases: int = 3,
) -> dict[str, Any]:
    response = await critic.complete(
        [
            {"role": "system", "content": critic_system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_id": task.id,
                        "task_instruction": task.instruction,
                        "expected_answer": task.expected_answer,
                        "rubric": task.rubric,
                        "policy_name": policy_name,
                        "patch_bundles": [bundle.model_dump() for bundle in patch_bundles],
                        "existing_policy_tests": existing_policy_tests,
                        "max_cases": max_cases,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ],
        temperature=0.1,
    )
    parsed = parse_critic_audit(response)
    parsed["audit_cases"] = parsed.get("audit_cases", [])[:max_cases]
    return parsed


def validate_critic_policy_cases(repo_root: Path, policy_path: Path, audit_cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not audit_cases:
        return {"ok": True, "reason": "no critic audit cases", "policy": policy_path.stem, "num_cases": 0}
    data = {"policy": policy_path.stem, "cases": audit_cases}
    return validate_runtime_policy_case_data(repo_root, policy_path, data, reason="critic policy audit cases passed")
