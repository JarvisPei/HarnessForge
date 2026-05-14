from __future__ import annotations

from pathlib import Path


def load_system_prompt(
    path: Path,
    skills_dir: Path | None = None,
    guidelines_dir: Path | None = None,
    validators_dir: Path | None = None,
    tools_dir: Path | None = None,
) -> str:
    parts = [path.read_text().strip()]
    parts.extend(_load_markdown_dir("Harness guidelines", guidelines_dir))
    parts.extend(_load_markdown_dir("Available harness skills", skills_dir))
    parts.extend(_load_markdown_dir("Harness validators", validators_dir))
    parts.extend(_load_python_tool_specs(tools_dir))
    return "\n\n".join(parts)


def _load_markdown_dir(title: str, directory: Path | None) -> list[str]:
    if not directory or not directory.exists():
        return []
    entries = []
    for item_path in sorted(directory.glob("*.md")):
        entries.append(f"## {item_path.stem}\n{item_path.read_text().strip()}")
    if not entries:
        return []
    return [f"{title}:\n\n" + "\n\n".join(entries)]


def _load_python_tool_specs(directory: Path | None) -> list[str]:
    if not directory or not directory.exists():
        return []
    entries = []
    for item_path in sorted(directory.glob("*.py")):
        entries.append(f"## Tool module: {item_path.stem}\n```python\n{item_path.read_text().strip()}\n```")
    if not entries:
        return []
    return ["Harness tool specs:\n\n" + "\n\n".join(entries)]
