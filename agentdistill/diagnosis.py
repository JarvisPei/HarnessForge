from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Diagnosis(BaseModel):
    diagnosis: str
    failure_categories: list[str] = Field(default_factory=list)
    harness_patch: str
    patch_type: str
    regression_test: str
    confidence: float | None = None
    parse_status: str = "parsed"
    patch_bundle: "PatchBundle | None" = None


class PatchBundle(BaseModel):
    target_path: str
    action: str = "create_or_replace"
    content: str
    rationale: str


def parse_diagnosis(raw: str) -> Diagnosis:
    payload = _extract_json(raw)
    if payload is None:
        return Diagnosis(
            diagnosis="Teacher response could not be parsed as JSON.",
            failure_categories=["parser"],
            harness_patch=raw.strip(),
            patch_type="unparsed",
            regression_test="Teacher diagnosis should return a JSON object with the required fields.",
            confidence=None,
            parse_status="unparsed",
        )
    data = json.loads(payload)
    if "patch_type" not in data:
        data["patch_type"] = _infer_patch_type(data.get("failure_categories", []), data.get("harness_patch", ""))
    if "regression_test" not in data:
        data["regression_test"] = "Add a regression test covering the diagnosed failure mode."
    return Diagnosis.model_validate(data)


def write_patch_artifact(
    patch_dir: Path,
    task_id: str,
    profile: str,
    diagnosis: Diagnosis,
) -> Path:
    patch_dir.mkdir(parents=True, exist_ok=True)
    safe_task = re.sub(r"[^a-zA-Z0-9_.-]+", "-", task_id)
    path = patch_dir / f"{profile}-{safe_task}.md"
    content = [
        f"# Harness Patch: {task_id}",
        "",
        f"- profile: `{profile}`",
        f"- patch_type: `{diagnosis.patch_type}`",
        f"- failure_categories: `{', '.join(diagnosis.failure_categories) or 'none'}`",
        f"- confidence: `{diagnosis.confidence if diagnosis.confidence is not None else 'unknown'}`",
        f"- parse_status: `{diagnosis.parse_status}`",
        "",
        "## Diagnosis",
        "",
        diagnosis.diagnosis.strip(),
        "",
        "## Harness Patch",
        "",
        diagnosis.harness_patch.strip(),
        "",
        "## Regression Test",
        "",
        diagnosis.regression_test.strip(),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def apply_patch_bundle(repo_root: Path, bundle: PatchBundle) -> Path:
    if bundle.action != "create_or_replace":
        raise RuntimeError(f"Unsupported patch bundle action: {bundle.action}")
    target = _safe_harness_path(repo_root, bundle.target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(bundle.content.strip() + "\n", encoding="utf-8")
    return target


def _safe_harness_path(repo_root: Path, target_path: str) -> Path:
    target = (repo_root / target_path).resolve()
    allowed_roots = [
        (repo_root / "harness" / "guidelines").resolve(),
        (repo_root / "harness" / "skills").resolve(),
        (repo_root / "harness" / "validators").resolve(),
    ]
    if target.suffix != ".md":
        raise RuntimeError(f"Patch target must be a markdown file: {target_path}")
    if not any(target.is_relative_to(root) for root in allowed_roots):
        raise RuntimeError(f"Patch target is outside allowed harness directories: {target_path}")
    return target


def _extract_json(raw: str) -> str | None:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    return None


def _infer_patch_type(categories: list[Any], patch: str) -> str:
    allowed = {
        "prompt_guideline",
        "skill",
        "tool",
        "validator",
        "state_representation",
        "runtime_policy",
    }
    for category in categories:
        if category in allowed:
            return str(category)
    lowered = patch.lower()
    for candidate in ["validator", "tool", "skill"]:
        if candidate in lowered:
            return candidate
    return "prompt_guideline"
