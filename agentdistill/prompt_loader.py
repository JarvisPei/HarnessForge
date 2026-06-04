from __future__ import annotations

import os
from pathlib import Path


def load_teacher_system_prompt(repo_root: Path) -> str:
    prompt_path = os.getenv("TEACHER_PROMPT_PATH")
    path = Path(prompt_path) if prompt_path else repo_root / "prompts" / "teacher_diagnosis.md"
    if not path.is_absolute():
        path = repo_root / path
    return path.read_text(encoding="utf-8").strip()
