from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from agentdistill.config import ExperimentConfig, TaskConfig, load_config
from agentdistill.contracts import validate_runtime_policy_contract, validate_tool_contract
from agentdistill.diagnosis import apply_patch_bundle, parse_diagnosis, write_patch_artifact
from agentdistill.harness import load_system_prompt
from agentdistill.models import ChatClient, load_model_settings
from agentdistill.tools import RuntimePolicyRegistry, ToolRegistry


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c"),
    profile: str | None = typer.Option(None, "--profile", "-p"),
    apply_patches: bool = typer.Option(False, "--apply-patches"),
    apply_success_patches: bool = typer.Option(False, "--apply-success-patches"),
    iterations: int = typer.Option(1, "--iterations", min=1),
) -> None:
    load_dotenv(override=True)
    cfg = load_config(config)
    try:
        asyncio.run(run_experiment(cfg, profile, apply_patches, apply_success_patches, iterations))
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


async def run_experiment(
    cfg: ExperimentConfig,
    profile: str | None,
    apply_patches: bool,
    apply_success_patches: bool,
    iterations: int,
) -> None:
    output_dir = cfg.output_dir / profile.lower() if profile else cfg.output_dir / "default"
    output_dir.mkdir(parents=True, exist_ok=True)
    weak = ChatClient(load_model_settings("weak", profile))
    teacher = ChatClient(load_model_settings("teacher", profile))
    repo_root = Path(__file__).resolve().parent.parent
    teacher_system = (repo_root / "prompts/teacher_diagnosis.md").read_text().strip()

    console.print(f"[bold]Experiment:[/bold] {cfg.name}")
    if profile:
        console.print(f"[bold]Profile:[/bold] {profile.upper()}")
    console.print(f"[bold]Output:[/bold] {output_dir}")

    summary: list[dict[str, object]] = []
    for iteration in range(1, iterations + 1):
        console.print(f"\n[bold]Iteration:[/bold] {iteration}/{iterations}")
        iteration_dir = output_dir / f"iter_{iteration:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        for task in cfg.tasks:
            console.print(f"\n[bold cyan]Task[/bold cyan] {task.id}")
            tools = ToolRegistry(cfg.harness.tools_dir)
            policies = RuntimePolicyRegistry(cfg.harness.runtime_policies_dir)
            weak_system = load_system_prompt(
                cfg.harness.system_prompt_path,
                cfg.harness.skills_dir,
                cfg.harness.guidelines_dir,
                cfg.harness.validators_dir,
                cfg.harness.tools_dir,
            )
            if tools.names:
                weak_system = weak_system + "\n\n" + tools.describe()
            result = await run_task(task, weak, teacher, weak_system, teacher_system, tools, policies)
            diagnosis = parse_diagnosis(str(result["teacher_diagnosis_raw"]))
            result["teacher_diagnosis"] = diagnosis.model_dump()
            applied_patch_path = None
            if (
                apply_patches
                and diagnosis.patch_bundle is not None
                and (apply_success_patches or bool(diagnosis.failure_categories))
            ):
                applied_patch_path = apply_patch_bundle(repo_root, diagnosis.patch_bundle)
                result["applied_patch_path"] = str(applied_patch_path)
                if applied_patch_path.is_relative_to((repo_root / "harness" / "tools").resolve()):
                    contract = validate_tool_contract(repo_root, applied_patch_path)
                    result["contract_validation"] = contract
                    if contract.get("ok") is not True:
                        applied_patch_path.unlink(missing_ok=True)
                        result["rejected_patch_path"] = str(applied_patch_path)
                        result["patch_status"] = "rejected"
                        result["applied_patch_path"] = None
                        applied_patch_path = None
                    else:
                        result["patch_status"] = "accepted"
                if applied_patch_path.is_relative_to((repo_root / "harness" / "runtime_policies").resolve()):
                    contract = validate_runtime_policy_contract(repo_root, task, applied_patch_path)
                    result["contract_validation"] = contract
                    if contract.get("ok") is not True:
                        applied_patch_path.unlink(missing_ok=True)
                        result["rejected_patch_path"] = str(applied_patch_path)
                        result["patch_status"] = "rejected"
                        result["applied_patch_path"] = None
                        applied_patch_path = None
                    else:
                        result["patch_status"] = "accepted"
            output_path = iteration_dir / f"{task.id}.json"
            output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            patch_path = write_patch_artifact(iteration_dir / "patches", task.id, profile or "default", diagnosis)
            console.print(f"Saved {output_path}")
            console.print(f"Patch {patch_path}")
            if applied_patch_path is not None:
                console.print(f"Applied {applied_patch_path}")
            summary.append(
                {
                    "iteration": iteration,
                    "task_id": task.id,
                    "patch_type": diagnosis.patch_type,
                    "failure_categories": diagnosis.failure_categories,
                    "parse_status": diagnosis.parse_status,
                    "confidence": diagnosis.confidence,
                    "output_path": str(output_path),
                    "patch_path": str(patch_path),
                    "applied_patch_path": str(applied_patch_path) if applied_patch_path else None,
                }
            )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    console.print(f"\n[bold]Summary:[/bold] {summary_path}")


async def run_task(
    task: TaskConfig,
    weak: ChatClient,
    teacher: ChatClient,
    weak_system: str,
    teacher_system: str,
    tools: ToolRegistry,
    policies: RuntimePolicyRegistry,
) -> dict[str, object]:
    weak_messages = [
        {"role": "system", "content": weak_system},
        {"role": "user", "content": task.instruction},
    ]
    weak_answer = await weak.complete(weak_messages)
    tool_call = _parse_tool_call(weak_answer)
    tool_result = None
    policy_results: list[dict[str, object]] = []
    final_answer = weak_answer
    if tool_call is not None:
        tool_result = tools.call(tool_call["name"], tool_call.get("input", {}))
        final_answer = await weak.complete(
            [
                {"role": "system", "content": weak_system},
                {"role": "user", "content": task.instruction},
                {"role": "assistant", "content": weak_answer},
                {
                    "role": "user",
                    "content": "Tool result:\n"
                    + json.dumps(tool_result, ensure_ascii=False, indent=2)
                    + "\n\nNow answer the original task directly.",
                },
            ]
        )
    else:
        policy_results = policies.evaluate(
            {
                "task_instruction": task.instruction,
                "initial_answer": weak_answer,
                "tool_call": None,
                "available_tools": tools.names,
                "expected_answer": task.expected_answer,
                "rubric": task.rubric,
            }
        )
        forced = _first_forced_tool(policy_results)
        if forced is not None:
            forced_tool_call = {"name": forced["tool_name"], "input": forced.get("tool_input", {})}
            tool_call = forced_tool_call
            tool_result = tools.call(str(forced_tool_call["name"]), forced_tool_call.get("input", {}))
            final_answer = await weak.complete(
                [
                    {"role": "system", "content": weak_system},
                    {"role": "user", "content": task.instruction},
                    {"role": "assistant", "content": weak_answer},
                    {
                        "role": "user",
                        "content": "A runtime policy required this tool call before finalizing:\n"
                        + json.dumps(forced_tool_call, ensure_ascii=False, indent=2)
                        + "\n\nTool result:\n"
                        + json.dumps(tool_result, ensure_ascii=False, indent=2)
                        + "\n\nNow answer the original task directly using the tool result.",
                    },
                ]
            )

    teacher_messages = [
        {"role": "system", "content": teacher_system},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task_id": task.id,
                    "task_instruction": task.instruction,
                    "expected_answer": task.expected_answer,
                    "rubric": task.rubric,
                    "weak_system_prompt": weak_system,
                    "weak_answer": final_answer,
                    "initial_weak_answer": weak_answer,
                    "tool_call": tool_call,
                    "tool_result": tool_result,
                    "runtime_policy_results": policy_results,
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
        "weak_answer": final_answer,
        "initial_weak_answer": weak_answer,
        "tool_call": tool_call,
        "tool_result": tool_result,
        "runtime_policy_results": policy_results,
        "teacher_diagnosis_raw": diagnosis,
    }


def _parse_tool_call(content: str) -> dict[str, object] | None:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    tool_call = payload.get("tool_call")
    if not isinstance(tool_call, dict):
        return None
    name = tool_call.get("name")
    input_payload = tool_call.get("input", {})
    if not isinstance(name, str) or not isinstance(input_payload, dict):
        return None
    return {"name": name, "input": input_payload}


def _first_forced_tool(policy_results: list[dict[str, object]]) -> dict[str, object] | None:
    for result in policy_results:
        if result.get("requires_tool") is True and isinstance(result.get("tool_name"), str):
            return result
    return None


if __name__ == "__main__":
    app()
