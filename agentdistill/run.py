from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from agentdistill.config import ExperimentConfig, TaskConfig, load_config
from agentdistill.harness import load_system_prompt
from agentdistill.models import ChatClient, load_model_settings


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(config: Path = typer.Option(..., "--config", "-c")) -> None:
    load_dotenv()
    cfg = load_config(config)
    try:
        asyncio.run(run_experiment(cfg))
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


async def run_experiment(cfg: ExperimentConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    weak = ChatClient(load_model_settings("weak"))
    teacher = ChatClient(load_model_settings("teacher"))
    weak_system = load_system_prompt(cfg.harness.system_prompt_path, cfg.harness.skills_dir)
    repo_root = Path(__file__).resolve().parent.parent
    teacher_system = (repo_root / "prompts/teacher_diagnosis.md").read_text().strip()

    console.print(f"[bold]Experiment:[/bold] {cfg.name}")
    console.print(f"[bold]Output:[/bold] {cfg.output_dir}")

    for task in cfg.tasks:
        console.print(f"\n[bold cyan]Task[/bold cyan] {task.id}")
        result = await run_task(task, weak, teacher, weak_system, teacher_system)
        output_path = cfg.output_dir / f"{task.id}.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        console.print(f"Saved {output_path}")


async def run_task(
    task: TaskConfig,
    weak: ChatClient,
    teacher: ChatClient,
    weak_system: str,
    teacher_system: str,
) -> dict[str, object]:
    weak_messages = [
        {"role": "system", "content": weak_system},
        {"role": "user", "content": task.instruction},
    ]
    weak_answer = await weak.complete(weak_messages)

    teacher_messages = [
        {"role": "system", "content": teacher_system},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task_id": task.id,
                    "task_instruction": task.instruction,
                    "weak_system_prompt": weak_system,
                    "weak_answer": weak_answer,
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]
    diagnosis = await teacher.complete(teacher_messages, temperature=0.1)

    return {
        "task_id": task.id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weak_answer": weak_answer,
        "teacher_diagnosis_raw": diagnosis,
    }


if __name__ == "__main__":
    app()
