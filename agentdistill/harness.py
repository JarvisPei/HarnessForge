from __future__ import annotations

from pathlib import Path


def load_system_prompt(path: Path, skills_dir: Path | None = None) -> str:
    parts = [path.read_text().strip()]
    if skills_dir and skills_dir.exists():
        skill_texts = []
        for skill_path in sorted(skills_dir.glob("*.md")):
            skill_texts.append(f"## Skill: {skill_path.stem}\n{skill_path.read_text().strip()}")
        if skill_texts:
            parts.append("Available harness skills:\n\n" + "\n\n".join(skill_texts))
    return "\n\n".join(parts)
