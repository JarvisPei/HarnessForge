from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentdistill.config import TaskConfig
from agentdistill.contracts import validate_runtime_policy_contract, validate_tool_contract
from agentdistill.diagnosis import PatchBundle, _safe_harness_path, apply_patch_bundle


@dataclass(frozen=True)
class AppliedPatch:
    path: Path
    previous_content: str | None


def apply_patch_bundles_atomically(
    repo_root: Path,
    bundles: list[PatchBundle],
    task: TaskConfig,
) -> dict[str, Any]:
    applied: list[AppliedPatch] = []
    contract_results: list[dict[str, Any]] = []
    try:
        for bundle in bundles:
            target = _safe_harness_path(repo_root, bundle.target_path)
            previous = target.read_text(encoding="utf-8") if target.exists() else None
            applied.append(AppliedPatch(path=target, previous_content=previous))
            path = apply_patch_bundle(repo_root, bundle)
            applied[-1] = AppliedPatch(path=path, previous_content=previous)

        contract_results = _validate_patch_group(repo_root, task, [patch.path for patch in applied])
        failures = [result for result in contract_results if result.get("ok") is not True]
        if failures:
            raise PatchGroupRejected("one or more patch contracts failed", contract_results)

        return {
            "patch_status": "accepted",
            "applied_patch_paths": [str(patch.path) for patch in applied],
            "rejected_patch_paths": [],
            "contract_validation": contract_results,
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
            "rejected_patch_paths": [str(patch.path) for patch in applied],
            "contract_validation": contract_results,
            "rejection_reason": reason,
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
    return results


def _rollback(applied: list[AppliedPatch]) -> None:
    for patch in reversed(applied):
        if patch.previous_content is None:
            patch.path.unlink(missing_ok=True)
        else:
            patch.path.write_text(patch.previous_content, encoding="utf-8")
