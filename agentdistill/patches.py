from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentdistill.config import TaskConfig
from agentdistill.contracts import validate_runtime_policy_contract, validate_runtime_policy_tests, validate_tool_contract
from agentdistill.diagnosis import PatchBundle, _safe_harness_path, apply_patch_bundle
from agentdistill.manifest import HarnessManifest, validate_harness_manifest


@dataclass(frozen=True)
class AppliedPatch:
    path: Path
    previous_content: str | None


def apply_patch_bundles_atomically(
    repo_root: Path,
    bundles: list[PatchBundle],
    task: TaskConfig,
    manifest: HarnessManifest | None = None,
) -> dict[str, Any]:
    applied: list[AppliedPatch] = []
    attempted_paths: list[Path] = []
    contract_results: list[dict[str, Any]] = []
    try:
        attempted_paths = [_safe_harness_path(repo_root, bundle.target_path) for bundle in bundles]
        manifest_results = validate_harness_manifest(repo_root, bundles, manifest)
        manifest_failures = [result for result in manifest_results if result.get("ok") is not True]
        if manifest_failures:
            raise PatchGroupRejected("harness manifest validation failed", manifest_results)

        staging_root = _staging_root(repo_root, manifest)
        _copy_harness_tree(repo_root, staging_root)
        staged_applied: list[AppliedPatch] = []
        for bundle in bundles:
            target = _safe_harness_path(staging_root, bundle.target_path)
            previous = target.read_text(encoding="utf-8") if target.exists() else None
            staged_applied.append(AppliedPatch(path=target, previous_content=previous))
            path = apply_patch_bundle(staging_root, bundle)
            staged_applied[-1] = AppliedPatch(path=path, previous_content=previous)

        contract_results = manifest_results + _validate_patch_group(staging_root, task, [patch.path for patch in staged_applied])
        failures = [result for result in contract_results if result.get("ok") is not True]
        if failures:
            raise PatchGroupRejected("one or more patch contracts failed", contract_results)

        for staged_patch in staged_applied:
            relative = staged_patch.path.relative_to(staging_root)
            final_path = (repo_root / relative).resolve()
            previous = final_path.read_text(encoding="utf-8") if final_path.exists() else None
            applied.append(AppliedPatch(path=final_path, previous_content=previous))
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_text(staged_patch.path.read_text(encoding="utf-8"), encoding="utf-8")

        return {
            "patch_status": "accepted",
            "applied_patch_paths": [str(patch.path) for patch in applied],
            "rejected_patch_paths": [],
            "contract_validation": contract_results,
            "harness_manifest": manifest.model_dump() if manifest is not None else None,
        }
    except Exception as exc:
        _rollback(applied)
        reason = str(exc)
        if isinstance(exc, PatchGroupRejected):
            contract_results = exc.contract_results
            reason = exc.reason
        return {
            "patch_status": "rejected",
            "applied_patch_paths": [],
            "rejected_patch_paths": [str(path) for path in attempted_paths],
            "contract_validation": contract_results,
            "rejection_reason": reason,
            "harness_manifest": manifest.model_dump() if manifest is not None else None,
        }


class PatchGroupRejected(RuntimeError):
    def __init__(self, reason: str, contract_results: list[dict[str, Any]]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.contract_results = contract_results


def _validate_patch_group(repo_root: Path, task: TaskConfig, paths: list[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    tools_root = (repo_root / "harness" / "tools").resolve()
    policies_root = (repo_root / "harness" / "runtime_policies").resolve()
    for path in paths:
        if path.is_relative_to(tools_root):
            results.append({"path": str(path), **validate_tool_contract(repo_root, path)})
        if path.is_relative_to(policies_root):
            results.append({"path": str(path), **validate_runtime_policy_contract(repo_root, task, path)})
            results.append({"path": str(path), **validate_runtime_policy_tests(repo_root, path)})
    return results


def _rollback(applied: list[AppliedPatch]) -> None:
    for patch in reversed(applied):
        if patch.previous_content is None:
            patch.path.unlink(missing_ok=True)
        else:
            patch.path.write_text(patch.previous_content, encoding="utf-8")


def _staging_root(repo_root: Path, manifest: HarnessManifest | None) -> Path:
    suffix = manifest.bundle_id if manifest is not None and manifest.bundle_id else "anonymous"
    return repo_root / "outputs" / "harness_workspaces" / suffix


def _copy_harness_tree(repo_root: Path, staging_root: Path) -> None:
    if staging_root.exists():
        _remove_tree(staging_root)
    for subdir in ["guidelines", "skills", "validators", "tools", "runtime_policies", "tests"]:
        source = repo_root / "harness" / subdir
        target = staging_root / "harness" / subdir
        target.mkdir(parents=True, exist_ok=True)
        if source.exists():
            for path in source.glob("**/*"):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    relative = path.relative_to(source)
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _remove_tree(path: Path) -> None:
    for child in sorted(path.glob("**/*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()
