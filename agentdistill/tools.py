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
                validate_python_harness_file(path, required_function="run")
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
        module = _load_module(self._tools[name], f"agentdistill_tool_{name}", required_function="run")
        result = module.run(payload)
        json.dumps(result)
        return result


def _load_module(path: Path, name: str, required_function: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load tool module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, required_function):
        raise RuntimeError(f"Module {path} has no {required_function} function")
    return module


def validate_python_harness_file(path: Path, required_function: str) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    banned_calls = {"eval", "exec", "open", "__import__"}
    banned_import_roots = {"os", "subprocess", "socket", "pathlib", "shutil"}
    has_required = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in banned_import_roots:
                    raise RuntimeError(f"{path} imports banned module: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in banned_import_roots:
                raise RuntimeError(f"{path} imports banned module: {node.module}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in banned_calls:
            raise RuntimeError(f"{path} calls banned function: {node.func.id}")
        if isinstance(node, ast.FunctionDef) and node.name == required_function:
            has_required = True
    if not has_required:
        raise RuntimeError(f"{path} must define {required_function}(input: dict) -> dict")


class RuntimePolicyRegistry:
    def __init__(self, policies_dir: Path | None):
        self.policies_dir = policies_dir
        self._policies: dict[str, Path] = {}
        if policies_dir and policies_dir.exists():
            for path in sorted(policies_dir.glob("*.py")):
                validate_python_harness_file(path, required_function="evaluate")
                self._policies[path.stem] = path

    @property
    def names(self) -> list[str]:
        return sorted(self._policies)

    def evaluate(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        results = []
        for name, path in self._policies.items():
            module = _load_module(path, f"agentdistill_policy_{name}", required_function="evaluate")
            result = module.evaluate(payload)
            json.dumps(result)
            if isinstance(result, dict):
                result = {"policy": name, **result}
                results.append(result)
        return results
