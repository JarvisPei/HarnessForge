from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


CODE_ARTIFACT_DIRS = {"tools", "runtime_policies", "tests"}
KNOWN_ARTIFACT_TYPES = {
    "guideline",
    "skill",
    "validator",
    "tool",
    "runtime_policy",
    "test",
}


class ManifestArtifact(BaseModel):
    path: str
    type: str
    purpose: str = ""


class HarnessManifest(BaseModel):
    bundle_id: str = ""
    intent: str = ""
    allowed_paths: list[str] = Field(default_factory=list)
    artifacts: list[ManifestArtifact] = Field(default_factory=list)
    contracts: list[str] = Field(default_factory=list)


def validate_harness_manifest(
    repo_root: Path,
    bundles: list[Any],
    manifest: HarnessManifest | None,
) -> list[dict[str, Any]]:
    if manifest is None:
        if _requires_manifest(bundles):
            return [{"ok": False, "reason": "code harness bundles must include harness_manifest"}]
        return [{"ok": True, "reason": "manifest optional for prompt-only bundle"}]

    results: list[dict[str, Any]] = []
    if not manifest.bundle_id or not re.fullmatch(r"[a-zA-Z0-9_.-]+", manifest.bundle_id):
        results.append({"ok": False, "reason": "manifest bundle_id must be a non-empty safe identifier"})
    if not manifest.intent.strip():
        results.append({"ok": False, "reason": "manifest intent must explain the harness change"})

    bundle_paths = [_normalized_harness_path(repo_root, bundle.target_path) for bundle in bundles]
    allowed_paths = [_normalized_harness_path(repo_root, path) for path in manifest.allowed_paths]
    artifact_paths = [_normalized_harness_path(repo_root, artifact.path) for artifact in manifest.artifacts]

    missing_allowed = sorted(set(bundle_paths) - set(allowed_paths))
    if missing_allowed:
        results.append({"ok": False, "reason": "manifest allowed_paths missing bundle targets", "paths": missing_allowed})

    missing_artifacts = sorted(set(bundle_paths) - set(artifact_paths))
    if missing_artifacts:
        results.append({"ok": False, "reason": "manifest artifacts missing bundle targets", "paths": missing_artifacts})

    extra_artifacts = sorted(set(artifact_paths) - set(bundle_paths))
    if extra_artifacts:
        results.append({"ok": False, "reason": "manifest artifacts include paths not in patch_bundles", "paths": extra_artifacts})

    for artifact in manifest.artifacts:
        artifact_type = artifact.type
        if artifact_type not in KNOWN_ARTIFACT_TYPES:
            results.append({"ok": False, "reason": "manifest artifact has unknown type", "artifact": artifact.model_dump()})
            continue
        path_type = _artifact_type_for_path(repo_root, artifact.path)
        if path_type != artifact_type:
            results.append(
                {
                    "ok": False,
                    "reason": "manifest artifact type does not match path",
                    "path": artifact.path,
                    "declared_type": artifact_type,
                    "path_type": path_type,
                }
            )

    if not manifest.contracts:
        results.append({"ok": False, "reason": "manifest contracts must list at least one validation expectation"})

    if not results:
        return [{"ok": True, "reason": "manifest matches patch bundle", "bundle_id": manifest.bundle_id}]
    return results


def _requires_manifest(bundles: list[PatchBundle]) -> bool:
    return any(_harness_subdir(bundle.target_path) in CODE_ARTIFACT_DIRS for bundle in bundles)


def _normalized_harness_path(repo_root: Path, target_path: str) -> str:
    return str(_safe_harness_path(repo_root, target_path).relative_to(repo_root))


def _artifact_type_for_path(repo_root: Path, target_path: str) -> str | None:
    subdir = _harness_subdir(str(_safe_harness_path(repo_root, target_path).relative_to(repo_root)))
    return {
        "guidelines": "guideline",
        "skills": "skill",
        "validators": "validator",
        "tools": "tool",
        "runtime_policies": "runtime_policy",
        "tests": "test",
    }.get(subdir)


def _harness_subdir(target_path: str) -> str | None:
    parts = Path(target_path).parts
    if len(parts) < 2 or parts[0] != "harness":
        return None
    return parts[1]


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
    if target.suffix == ".py":
        tools_root = (repo_root / "harness" / "tools").resolve()
        runtime_root = (repo_root / "harness" / "runtime_policies").resolve()
        if not target.is_relative_to(tools_root) and not target.is_relative_to(runtime_root):
            raise RuntimeError(f"Python patch targets are only allowed under harness/tools or harness/runtime_policies: {target_path}")
    if target.suffix == ".json" and not target.is_relative_to((repo_root / "harness" / "tests").resolve()):
        raise RuntimeError(f"JSON patch targets are only allowed under harness/tests: {target_path}")
    if target == (repo_root / "harness" / "guidelines" / "base.md").resolve():
        raise RuntimeError("Teacher patches may not replace harness/guidelines/base.md; create a focused guideline file instead.")
    if not any(target.is_relative_to(root) for root in allowed_roots):
        raise RuntimeError(f"Patch target is outside allowed harness directories: {target_path}")
    return target
