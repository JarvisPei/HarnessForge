from __future__ import annotations

import shutil
from pathlib import Path


HARNESS_SUBDIRS = ["guidelines", "skills", "validators", "tools", "runtime_policies"]


def snapshot_harness(repo_root: Path, destination: Path) -> None:
    harness_root = repo_root / "harness"
    destination.mkdir(parents=True, exist_ok=True)
    for subdir in HARNESS_SUBDIRS:
        source = harness_root / subdir
        target = destination / subdir
        if source.exists():
            shutil.copytree(source, target, dirs_exist_ok=True)


def list_harness_files(repo_root: Path) -> list[str]:
    harness_root = repo_root / "harness"
    if not harness_root.exists():
        return []
    return sorted(str(path.relative_to(repo_root)) for path in harness_root.glob("**/*") if path.is_file())
