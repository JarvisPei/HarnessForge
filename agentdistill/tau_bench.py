from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console

from agentdistill.harness import load_system_prompt
from agentdistill.models import ChatClient, load_model_settings


app = typer.Typer(add_completion=False)
console = Console()


@dataclass(frozen=True)
class TauHarnessSettings:
    repo_root: Path
    profile: str | None = None
    system_prompt_path: Path | None = None
    skills_dir: Path | None = None
    guidelines_dir: Path | None = None
    validators_dir: Path | None = None
    max_tool_specs_chars: int = 12000


@dataclass
class HarnessForgeTauState:
    messages: list[Any] = field(default_factory=list)


@app.command()
def smoke(
    domain: str = typer.Option("airline", "--domain"),
    split: str = typer.Option("train", "--split"),
    num_tasks: int = typer.Option(2, "--num-tasks", min=1),
    task_set_name: str | None = typer.Option(None, "--task-set-name"),
    user_llm: str | None = typer.Option(None, "--user-llm"),
    profile: str | None = typer.Option(None, "--profile", "-p"),
    output_dir: Path = typer.Option(Path("outputs/tau_bench_smoke"), "--output-dir"),
    max_steps: int = typer.Option(80, "--max-steps", min=1),
    max_errors: int = typer.Option(5, "--max-errors", min=1),
    timeout: float | None = typer.Option(600.0, "--timeout"),
    seed: int = typer.Option(300, "--seed"),
) -> None:
    """Run a tiny tau2-bench text smoke with HarnessForge as the weak agent."""

    load_dotenv(override=True)
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = (repo_root / output_dir).resolve() if not output_dir.is_absolute() else output_dir
    settings = TauHarnessSettings(repo_root=repo_root, profile=profile)
    try:
        traces = run_tau_bench_smoke(
            settings=settings,
            domain=domain,
            split=split,
            num_tasks=num_tasks,
            task_set_name=task_set_name,
            user_llm=user_llm or os.getenv("TAU_USER_LLM", "gpt-4.1"),
            output_dir=output_dir,
            max_steps=max_steps,
            max_errors=max_errors,
            timeout=timeout,
            seed=seed,
        )
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold]tau-bench smoke traces:[/bold] {output_dir}")
    console.print(f"[bold]tasks:[/bold] {len(traces)}")


def run_tau_bench_smoke(
    *,
    settings: TauHarnessSettings,
    domain: str,
    split: str,
    num_tasks: int,
    task_set_name: str | None,
    user_llm: str,
    output_dir: Path,
    max_steps: int,
    max_errors: int,
    timeout: float | None,
    seed: int,
) -> list[dict[str, Any]]:
    tau = _load_tau2_runtime()
    agent_name = "harnessforge_agent"
    _register_harnessforge_agent(tau, agent_name, settings)
    tasks = _load_tau_tasks(tau, domain=domain, split=split, task_set_name=task_set_name, num_tasks=num_tasks)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = tau["TextRunConfig"](
        domain=domain,
        task_set_name=task_set_name,
        task_split_name=split,
        agent=agent_name,
        llm_agent="harnessforge-weak",
        user="user_simulator",
        llm_user=user_llm,
        num_tasks=len(tasks),
        max_steps=max_steps,
        max_errors=max_errors,
        timeout=timeout,
        seed=seed,
        enforce_communication_protocol=True,
    )

    traces: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        console.print(f"[cyan]tau task[/cyan] {index}/{len(tasks)} {task.id}")
        try:
            orchestrator = tau["build_text_orchestrator"](config, task, seed=seed + index)
            simulation_run = tau["run_simulation"](orchestrator)
            trace = simulation_run_to_trace(simulation_run, domain=domain, split=split)
        except Exception as exc:
            trace = task_error_to_trace(task, domain=domain, split=split, exc=exc)
            console.print(f"[red]tau task error[/red] {task.id}: {exc.__class__.__name__}: {exc}")
        traces.append(trace)
        (output_dir / f"{_safe_task_id(task.id)}.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "split": split,
        "task_set_name": task_set_name,
        "num_tasks": len(tasks),
        "agent": agent_name,
        "user_llm": user_llm,
        "traces": [
            {
                "task_id": trace.get("task_id"),
                "termination_reason": trace.get("termination_reason"),
                "reward_info": trace.get("reward_info"),
                "path": f"{_safe_task_id(str(trace.get('task_id')))}.json",
            }
            for trace in traces
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return traces


def _load_tau2_runtime() -> dict[str, Any]:
    try:
        from tau2 import TextRunConfig
        from tau2.agent.base_agent import HalfDuplexAgent, is_valid_agent_history_message
        from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall
        from tau2.registry import registry
        from tau2.runner import build_text_orchestrator, run_simulation
    except ImportError as exc:
        raise RuntimeError(
            "tau2-bench is not installed. Install it on the runtime server before running "
            "`python -m agentdistill.tau_bench smoke`."
        ) from exc
    return {
        "TextRunConfig": TextRunConfig,
        "HalfDuplexAgent": HalfDuplexAgent,
        "AssistantMessage": AssistantMessage,
        "MultiToolMessage": MultiToolMessage,
        "ToolCall": ToolCall,
        "registry": registry,
        "build_text_orchestrator": build_text_orchestrator,
        "run_simulation": run_simulation,
        "is_valid_agent_history_message": is_valid_agent_history_message,
    }


def _register_harnessforge_agent(tau: dict[str, Any], name: str, settings: TauHarnessSettings) -> None:
    registry = tau["registry"]
    if registry.get_agent_factory(name) is not None:
        return

    HalfDuplexAgent = tau["HalfDuplexAgent"]
    AssistantMessage = tau["AssistantMessage"]
    ToolCall = tau["ToolCall"]
    is_valid_agent_history_message = tau["is_valid_agent_history_message"]

    class HarnessForgeTauAgent(HalfDuplexAgent):
        def __init__(self, tools: list[Any], domain_policy: str, **_: Any):
            super().__init__(tools=tools, domain_policy=domain_policy)
            self.client = ChatClient(load_model_settings("weak", settings.profile))
            self.weak_system = build_tau_weak_system_prompt(
                domain_policy=domain_policy,
                tools=tools,
                settings=settings,
            )

        def get_init_state(self, message_history: list[Any] | None = None) -> HarnessForgeTauState:
            message_history = list(message_history or [])
            assert all(is_valid_agent_history_message(message) for message in message_history)
            return HarnessForgeTauState(messages=message_history)

        def generate_next_message(self, message: Any, state: HarnessForgeTauState) -> tuple[Any, HarnessForgeTauState]:
            if message is not None:
                if isinstance(message, tau["MultiToolMessage"]):
                    state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
            weak_messages = build_tau_weak_messages(self.weak_system, state.messages)
            raw = _complete_sync(self.client, weak_messages)
            if not raw.strip():
                raw = "I need a moment to review the information before continuing."
            tool_payloads = parse_tau_tool_calls(raw)
            if tool_payloads:
                assistant = AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=str(uuid.uuid4()),
                            name=payload["name"],
                            arguments=payload["arguments"],
                            requestor="assistant",
                        )
                        for payload in tool_payloads
                    ],
                    raw_data={"harnessforge_raw": raw},
                )
            else:
                assistant = AssistantMessage.text(raw, raw_data={"harnessforge_raw": raw})
            state.messages.append(assistant)
            return assistant, state

    def factory(tools: list[Any], domain_policy: str, **kwargs: Any) -> Any:
        return HarnessForgeTauAgent(tools=tools, domain_policy=domain_policy, **kwargs)

    registry.register_agent_factory(factory, name)


def build_tau_weak_system_prompt(*, domain_policy: str, tools: list[Any], settings: TauHarnessSettings) -> str:
    harness_prompt = load_system_prompt(
        settings.system_prompt_path or settings.repo_root / "prompts/weak_system.md",
        settings.skills_dir or settings.repo_root / "harness/skills",
        settings.guidelines_dir or settings.repo_root / "harness/guidelines",
        settings.validators_dir or settings.repo_root / "harness/validators",
        tools_dir=None,
    )
    tool_specs = describe_tau_tools(tools, max_chars=settings.max_tool_specs_chars)
    return "\n\n".join(
        [
            harness_prompt,
            "You are now operating inside tau-bench as the customer-service agent under test.",
            "Follow the domain policy exactly. The benchmark environment, user simulator, and evaluator are external and must be treated as authoritative.",
            "<domain_policy>\n" + domain_policy.strip() + "\n</domain_policy>",
            tool_specs,
            _TAU_COMMUNICATION_PROTOCOL,
        ]
    )


def describe_tau_tools(tools: list[Any], *, max_chars: int = 12000) -> str:
    entries = []
    for tool in tools:
        name = getattr(tool, "name", tool.__class__.__name__)
        schema = getattr(tool, "openai_schema", None)
        if schema is not None:
            description = json.dumps(schema, ensure_ascii=False, indent=2)
        elif hasattr(tool, "to_str"):
            description = str(tool.to_str())
        else:
            description = str(tool)
        entries.append(f"## {name}\n{description}")
    text = "Official tau-bench tools available to you:\n\n" + "\n\n".join(entries)
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[tool specs truncated]"
    return text


def build_tau_weak_messages(system_prompt: str, tau_messages: list[Any]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for message in tau_messages:
        converted = tau_message_to_chat_message(message)
        if converted is not None:
            messages.append(converted)
    return messages


def tau_message_to_chat_message(message: Any) -> dict[str, str] | None:
    role = getattr(message, "role", None)
    if role == "user":
        if getattr(message, "tool_calls", None):
            return {"role": "user", "content": _tool_calls_text(message.tool_calls, prefix="User tool call")}
        return {"role": "user", "content": str(getattr(message, "content", "") or "")}
    if role == "assistant":
        if getattr(message, "tool_calls", None):
            return {"role": "assistant", "content": _tool_calls_text(message.tool_calls, prefix="Assistant tool call")}
        return {"role": "assistant", "content": str(getattr(message, "content", "") or "")}
    if role == "tool":
        return {
            "role": "user",
            "content": "Tool result:\n"
            + json.dumps(
                {
                    "id": getattr(message, "id", None),
                    "requestor": getattr(message, "requestor", None),
                    "error": getattr(message, "error", None),
                    "content": getattr(message, "content", None),
                },
                ensure_ascii=False,
                indent=2,
            ),
        }
    tool_messages = getattr(message, "tool_messages", None)
    if tool_messages:
        return {
            "role": "user",
            "content": "Tool results:\n" + json.dumps([message_to_plain_dict(item) for item in tool_messages], ensure_ascii=False, indent=2),
        }
    return None


def parse_tau_tool_calls(content: str | None) -> list[dict[str, Any]]:
    if not content:
        return []
    text = _strip_code_fences(content.strip())
    if not text.startswith("{"):
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    candidates: list[Any] = []
    if isinstance(payload.get("tool_call"), dict):
        candidates = [payload["tool_call"]]
    elif isinstance(payload.get("tool_calls"), list):
        candidates = payload["tool_calls"]
    parsed = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        name = candidate.get("name")
        arguments = candidate.get("arguments", candidate.get("input", {}))
        if isinstance(name, str) and isinstance(arguments, dict):
            parsed.append({"name": name, "arguments": arguments})
    return parsed


def simulation_run_to_trace(simulation_run: Any, *, domain: str, split: str) -> dict[str, Any]:
    data = model_to_plain_data(simulation_run)
    messages = [message_to_plain_dict(message) for message in data.get("messages", [])]
    return {
        "schema": "harnessforge.tau_bench_trace.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "split": split,
        "task_id": data.get("task_id"),
        "simulation_id": data.get("id"),
        "termination_reason": data.get("termination_reason"),
        "reward_info": data.get("reward_info"),
        "agent_cost": data.get("agent_cost"),
        "user_cost": data.get("user_cost"),
        "duration": data.get("duration"),
        "seed": data.get("seed"),
        "messages": messages,
        "raw": data,
    }


def task_error_to_trace(task: Any, *, domain: str, split: str, exc: Exception) -> dict[str, Any]:
    return {
        "schema": "harnessforge.tau_bench_trace.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "split": split,
        "task_id": getattr(task, "id", None),
        "simulation_id": None,
        "termination_reason": "adapter_error",
        "reward_info": None,
        "agent_cost": None,
        "user_cost": None,
        "duration": None,
        "seed": None,
        "messages": [],
        "error": {
            "type": exc.__class__.__name__,
            "message": str(exc),
        },
        "raw": {},
    }


def model_to_plain_data(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Cannot serialize object of type {type(obj).__name__}")


def message_to_plain_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        data = dict(message)
    elif hasattr(message, "model_dump"):
        data = message.model_dump(mode="json")
    else:
        data = {
            "role": getattr(message, "role", None),
            "content": getattr(message, "content", None),
            "tool_calls": getattr(message, "tool_calls", None),
            "requestor": getattr(message, "requestor", None),
            "error": getattr(message, "error", None),
            "turn_idx": getattr(message, "turn_idx", None),
            "timestamp": getattr(message, "timestamp", None),
        }
    return {
        "role": data.get("role"),
        "content": data.get("content"),
        "tool_calls": data.get("tool_calls"),
        "requestor": data.get("requestor"),
        "error": data.get("error"),
        "turn_idx": data.get("turn_idx"),
        "timestamp": data.get("timestamp"),
    }


def _load_tau_tasks(tau: dict[str, Any], *, domain: str, split: str, task_set_name: str | None, num_tasks: int) -> list[Any]:
    registry = tau["registry"]
    loader_name = task_set_name or domain
    loader = registry.get_tasks_loader(loader_name)
    tasks = list(loader(split))
    if not tasks:
        raise RuntimeError(f"No tau-bench tasks loaded for task_set={loader_name} split={split}")
    return tasks[:num_tasks]


def _complete_sync(client: ChatClient, messages: list[dict[str, str]]) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return str(asyncio.run(client.complete(messages, temperature=0.1)) or "")
    raise RuntimeError("HarnessForge tau-bench adapter cannot run inside an existing asyncio loop yet")


def _tool_calls_text(tool_calls: list[Any], *, prefix: str) -> str:
    return prefix + ":\n" + json.dumps([_tool_call_to_dict(call) for call in tool_calls], ensure_ascii=False, indent=2)


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        return tool_call
    return {
        "id": getattr(tool_call, "id", None),
        "name": getattr(tool_call, "name", None),
        "arguments": getattr(tool_call, "arguments", None),
        "requestor": getattr(tool_call, "requestor", None),
    }


def _strip_code_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _safe_task_id(task_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in task_id)


_TAU_COMMUNICATION_PROTOCOL = """
Communication protocol:

- If you want to speak to the user, return only the user-facing message.
- If you need to call an official tau-bench tool, return only JSON in this form:
  {"tool_call": {"name": "tool_name", "arguments": {"arg": "value"}}}
- Do not mix a user-facing message and a tool call in the same turn.
- Do not invent helper tools. Only call tools listed in the official tau-bench tools section.
""".strip()


if __name__ == "__main__":
    app()
