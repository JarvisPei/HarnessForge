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


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    data: dict[str, Any] = yaml.safe_load(config_path.read_text())
    cfg = ExperimentConfig.model_validate(data)
    base = config_path.parent.parent
    cfg.output_dir = (base / cfg.output_dir).resolve()
    cfg.harness.system_prompt_path = (base / cfg.harness.system_prompt_path).resolve()
    if cfg.harness.skills_dir is not None:
        cfg.harness.skills_dir = (base / cfg.harness.skills_dir).resolve()
    if cfg.harness.guidelines_dir is not None:
        cfg.harness.guidelines_dir = (base / cfg.harness.guidelines_dir).resolve()
    if cfg.harness.validators_dir is not None:
        cfg.harness.validators_dir = (base / cfg.harness.validators_dir).resolve()
    if cfg.harness.tools_dir is not None:
        cfg.harness.tools_dir = (base / cfg.harness.tools_dir).resolve()
    return cfg
