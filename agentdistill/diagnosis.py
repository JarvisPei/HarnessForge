from __future__ import annotations

import json
import re
import py_compile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentdistill.tools import validate_python_harness_file


class Diagnosis(BaseModel):
    diagnosis: str
    failure_categories: list[str] = Field(default_factory=list)
    harness_patch: str
    patch_type: str
    regression_test: str
    confidence: float | None = None
    parse_status: str = "parsed"
    patch_bundle: "PatchBundle | None" = None
    patch_bundles: list["PatchBundle"] = Field(default_factory=list)


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
    elif not isinstance(data["regression_test"], str):
        data["regression_test"] = json.dumps(data["regression_test"], ensure_ascii=False)
    if "patch_bundles" not in data:
        data["patch_bundles"] = [data["patch_bundle"]] if data.get("patch_bundle") else []
    if "patch_bundle" not in data and data["patch_bundles"]:
        data["patch_bundle"] = data["patch_bundles"][0]
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
        "## Patch Bundles",
        "",
        json.dumps([bundle.model_dump() for bundle in diagnosis.patch_bundles], indent=2, ensure_ascii=False),
        "",
        "## Regression Test",
        "",
        diagnosis.regression_test.strip(),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def apply_patch_bundle(repo_root: Path, bundle: PatchBundle) -> Path:
    action = "create_or_replace" if bundle.action in {"create_or_replace", "replace"} else bundle.action
    if action != "create_or_replace":
        raise RuntimeError(f"Unsupported patch bundle action: {bundle.action}")
    target = _safe_harness_path(repo_root, bundle.target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(bundle.content.strip() + "\n", encoding="utf-8")
    if target.suffix == ".py":
        py_compile.compile(str(target), doraise=True)
        required = "evaluate" if target.is_relative_to((repo_root / "harness" / "runtime_policies").resolve()) else "run"
        validate_python_harness_file(target, required_function=required)
    return target


def _safe_harness_path(repo_root: Path, target_path: str) -> Path:
    target = (repo_root / target_path).resolve()
    allowed_roots = [
        (repo_root / "harness" / "guidelines").resolve(),
        (repo_root / "harness" / "skills").resolve(),
        (repo_root / "harness" / "validators").resolve(),
        (repo_root / "harness" / "tools").resolve(),
        (repo_root / "harness" / "runtime_policies").resolve(),
        (repo_root / "harness" / "tests").resolve(),
    ]
    if target.suffix not in {".md", ".py", ".json"}:
        raise RuntimeError(f"Patch target must be a markdown, Python, or JSON file: {target_path}")
    if target.suffix == ".py" and not target.is_relative_to((repo_root / "harness" / "tools").resolve()):
        runtime_root = (repo_root / "harness" / "runtime_policies").resolve()
        if not target.is_relative_to(runtime_root):
            raise RuntimeError(f"Python patch targets are only allowed under harness/tools or harness/runtime_policies: {target_path}")
    if target.suffix == ".json" and not target.is_relative_to((repo_root / "harness" / "tests").resolve()):
        raise RuntimeError(f"JSON patch targets are only allowed under harness/tests: {target_path}")
    if target == (repo_root / "harness" / "guidelines" / "base.md").resolve():
        raise RuntimeError("Teacher patches may not replace harness/guidelines/base.md; create a focused guideline file instead.")
    if not any(target.is_relative_to(root) for root in allowed_roots):
        raise RuntimeError(f"Patch target is outside allowed harness directories: {target_path}")
    return target


def _extract_json(raw: str) -> str | None:
    text = raw.strip()
    fenced = _extract_fenced_json(text)
    if fenced is not None:
        return fenced
    balanced = _extract_balanced_json(text)
    if balanced is not None:
        return balanced
    return None


def _extract_fenced_json(text: str) -> str | None:
    start = text.find("```")
    if start == -1:
        return None
    end = text.find("```", start + 3)
    if end == -1:
        return None
    block = text[start + 3 : end].strip()
    if block.startswith("json"):
        block = block[4:].lstrip()
    return _extract_balanced_json(block)


def _extract_balanced_json(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        candidate = _scan_balanced_object(text, start)
        if candidate is not None:
            return candidate
        start = text.find("{", start + 1)
    return None


def _scan_balanced_object(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
            if depth < 0:
                return None
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
