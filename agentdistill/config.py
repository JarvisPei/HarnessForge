from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RoleConfig(BaseModel):
    role: str


class HarnessConfig(BaseModel):
    system_prompt_path: Path
    skills_dir: Path | None = None
    guidelines_dir: Path | None = None
    validators_dir: Path | None = None
    tools_dir: Path | None = None
    runtime_policies_dir: Path | None = None


class TaskConfig(BaseModel):
    id: str
    instruction: str
    expected_answer: str | None = None
    rubric: str | None = None


class ExperimentConfig(BaseModel):
    name: str
    output_dir: Path = Field(default=Path("outputs/default"))
    weak: RoleConfig
    teacher: RoleConfig
    harness: HarnessConfig
    tasks: list[TaskConfig]


class BenchmarkConfig(BaseModel):
    name: str
    output_dir: Path
    weak: RoleConfig
    teacher: RoleConfig
    harness: HarnessConfig
    train_tasks: list[TaskConfig]
    heldout_tasks: list[TaskConfig] = Field(default_factory=list)
    dev_probe_tasks: list[TaskConfig] = Field(default_factory=list)
    blind_test_tasks: list[TaskConfig] = Field(default_factory=list)
    evolve_iterations: int = 2


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    data: dict[str, Any] = yaml.safe_load(config_path.read_text())
    cfg = ExperimentConfig.model_validate(data)
    base = config_path.parent.parent
    cfg.output_dir = (base / cfg.output_dir).resolve()
    _resolve_harness_paths(cfg.harness, base)
    return cfg


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    data: dict[str, Any] = yaml.safe_load(config_path.read_text())
    cfg = BenchmarkConfig.model_validate(data)
    if not cfg.dev_probe_tasks:
        cfg.dev_probe_tasks = list(cfg.heldout_tasks)
    if not cfg.blind_test_tasks:
        cfg.blind_test_tasks = list(cfg.heldout_tasks)
    base = config_path.parent.parent
    cfg.output_dir = (base / cfg.output_dir).resolve()
    _resolve_harness_paths(cfg.harness, base)
    return cfg


def _resolve_harness_paths(harness: HarnessConfig, base: Path) -> None:
    harness.system_prompt_path = (base / harness.system_prompt_path).resolve()
    if harness.skills_dir is not None:
        harness.skills_dir = (base / harness.skills_dir).resolve()
    if harness.guidelines_dir is not None:
        harness.guidelines_dir = (base / harness.guidelines_dir).resolve()
    if harness.validators_dir is not None:
        harness.validators_dir = (base / harness.validators_dir).resolve()
    if harness.tools_dir is not None:
        harness.tools_dir = (base / harness.tools_dir).resolve()
    if harness.runtime_policies_dir is not None:
        harness.runtime_policies_dir = (base / harness.runtime_policies_dir).resolve()
