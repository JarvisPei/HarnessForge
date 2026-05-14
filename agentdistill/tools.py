from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


class ToolRegistry:
    def __init__(self, tools_dir: Path | None):
        self.tools_dir = tools_dir
        self._tools: dict[str, Path] = {}
        if tools_dir and tools_dir.exists():
            for path in sorted(tools_dir.glob("*.py")):
                self._validate_tool_source(path)
                self._tools[path.stem] = path

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        if not self._tools:
            return "No callable tools are currently registered."
        return "Callable tools: " + ", ".join(f"`{name}`" for name in self.names)

    def call(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}", "available_tools": self.names}
        module = _load_module(self._tools[name], f"agentdistill_tool_{name}")
        result = module.run(payload)
        json.dumps(result)
        return result

    def _validate_tool_source(self, path: Path) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        banned_calls = {"eval", "exec", "open", "__import__"}
        banned_import_roots = {"os", "subprocess", "socket", "pathlib", "shutil"}
        has_run = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in banned_import_roots:
                        raise RuntimeError(f"Tool {path} imports banned module: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in banned_import_roots:
                    raise RuntimeError(f"Tool {path} imports banned module: {node.module}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in banned_calls:
                raise RuntimeError(f"Tool {path} calls banned function: {node.func.id}")
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                has_run = True
        if not has_run:
            raise RuntimeError(f"Tool {path} must define run(input: dict) -> dict")


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load tool module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise RuntimeError(f"Tool module has no run function: {path}")
    return module
