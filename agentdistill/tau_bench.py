from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import typer
from dotenv import load_dotenv
from rich.console import Console

from agentdistill.harness import load_system_prompt
from agentdistill.models import ChatClient, ModelSettings, load_model_settings
from agentdistill.tools import RuntimePolicyRegistry


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
    runtime_policies_dir: Path | None = None
    max_tool_specs_chars: int = 12000


@dataclass
class HarnessForgeTauState:
    messages: list[Any] = field(default_factory=list)


@app.command()
def smoke(
    domain: str = typer.Option("airline", "--domain"),
    split: str = typer.Option("train", "--split"),
    num_tasks: int = typer.Option(2, "--num-tasks", min=1),
    task_ids: list[str] | None = typer.Option(None, "--task-id"),
    task_set_name: str | None = typer.Option(None, "--task-set-name"),
    user_llm: str | None = typer.Option(None, "--user-llm"),
    profile: str | None = typer.Option(None, "--profile", "-p"),
    output_dir: Path = typer.Option(Path("outputs/tau_bench_smoke"), "--output-dir"),
    max_steps: int = typer.Option(80, "--max-steps", min=1),
    max_errors: int = typer.Option(5, "--max-errors", min=1),
    timeout: float | None = typer.Option(600.0, "--timeout"),
    seed: int = typer.Option(300, "--seed"),
    user_llm_shim: bool = typer.Option(
        False,
        "--user-llm-shim/--no-user-llm-shim",
        help="Route tau user-simulator text LLM calls through HarnessForge ChatClient instead of LiteLLM.",
    ),
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
            task_ids=task_ids,
            task_set_name=task_set_name,
            user_llm=user_llm or os.getenv("TAU_USER_LLM", "gpt-4.1"),
            output_dir=output_dir,
            max_steps=max_steps,
            max_errors=max_errors,
            timeout=timeout,
            seed=seed,
            user_llm_shim=user_llm_shim,
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
    task_ids: list[str] | None = None,
    user_llm_shim: bool = False,
) -> list[dict[str, Any]]:
    tau = _load_tau2_runtime()
    if user_llm_shim:
        _install_tau_user_llm_shim()
    agent_name = "harnessforge_agent"
    _register_harnessforge_agent(tau, agent_name, settings)
    tasks = _load_tau_tasks(
        tau,
        domain=domain,
        split=split,
        task_set_name=task_set_name,
        num_tasks=num_tasks,
        task_ids=task_ids,
    )
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
        "task_ids": [str(task.id) for task in tasks],
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


def _install_tau_user_llm_shim() -> None:
    """Patch tau2 text-only LiteLLM calls to use the HarnessForge ChatClient.

    Some OpenAI-compatible relays reject LiteLLM's request shape while accepting
    the raw chat-completions request used by ChatClient. The shim is opt-in and
    only handles text-only user-simulator calls; tool-enabled calls fall back to
    tau2's original LiteLLM completion function.
    """

    import tau2.utils.llm_utils as llm_utils

    if getattr(llm_utils, "_harnessforge_chat_client_shim", False):
        return

    original_completion = llm_utils.completion
    llm_utils.completion = _make_tau_user_llm_completion_shim(original_completion)
    llm_utils._harnessforge_chat_client_shim = True


def _make_tau_user_llm_completion_shim(original_completion: Any) -> Any:
    def completion_shim(
        *,
        model: str,
        messages: list[dict[str, str]],
        tools: list[Any] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if tools or tool_choice:
            return original_completion(model=model, messages=messages, tools=tools, tool_choice=tool_choice, **kwargs)
        settings = _tau_user_model_settings(str(model))
        temperature = float(kwargs.get("temperature", 0.1))
        text = _complete_sync(ChatClient(settings), messages, temperature=temperature)
        return _LiteLLMShimResponse(model=settings.model, content=text)

    return completion_shim


class _LiteLLMShimResponse(dict):
    def __init__(self, *, model: str, content: str):
        super().__init__({"model": model, "usage": None})
        self.model = model
        self.choices = [
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content=content, tool_calls=None),
            )
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": self.choices[0].message.content,
                        "tool_calls": None,
                    },
                }
            ],
            "usage": None,
        }


def _tau_user_model_settings(model: str) -> ModelSettings:
    base_url = (
        os.getenv("TAU_USER_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("TEACHER_BASE_URL")
    )
    api_key = os.getenv("TAU_USER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("TEACHER_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("TAU user LLM shim requires TAU_USER_* or OPENAI_* or TEACHER_* API env vars")
    return ModelSettings(
        role="teacher",
        provider="openai",
        base_url=base_url,
        api_key=api_key,
        model=model.removeprefix("openai/"),
        timeout_seconds=float(os.getenv("TAU_USER_TIMEOUT_SECONDS", os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))),
        max_retries=int(os.getenv("TAU_USER_MAX_RETRIES", os.getenv("REQUEST_MAX_RETRIES", "2"))),
        retry_backoff_seconds=float(
            os.getenv("TAU_USER_RETRY_BACKOFF_SECONDS", os.getenv("REQUEST_RETRY_BACKOFF_SECONDS", "2"))
        ),
    )


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
            self.policies = RuntimePolicyRegistry(
                settings.runtime_policies_dir or settings.repo_root / "harness" / "runtime_policies"
            )
            self.official_tool_names = sorted(
                {
                    name
                    for tool in tools
                    if isinstance((name := getattr(tool, "name", tool.__class__.__name__)), str)
                }
            )

        def get_init_state(self, message_history: list[Any] | None = None) -> HarnessForgeTauState:
            message_history = list(message_history or [])
            assert all(is_valid_agent_history_message(message) for message in message_history)
            return HarnessForgeTauState(messages=message_history)

        def generate_next_message(self, message: Any, state: HarnessForgeTauState) -> tuple[Any, HarnessForgeTauState]:
            _append_tau_incoming_message(state.messages, message)
            weak_messages = build_tau_weak_messages(self.weak_system, state.messages)
            raw = _complete_sync(self.client, weak_messages)
            if not raw.strip():
                raw = "I need a moment to review the information before continuing."
            tool_payloads, policy_results = _select_tau_tool_payloads_with_policy(
                raw=raw,
                tau_messages=state.messages,
                available_tools=self.official_tool_names,
                policies=self.policies,
            )
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
                    raw_data={"harnessforge_raw": raw, "runtime_policy_results": policy_results},
                )
            else:
                assistant = AssistantMessage.text(raw, raw_data={"harnessforge_raw": raw, "runtime_policy_results": policy_results})
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


def _append_tau_incoming_message(state_messages: list[Any], message: Any) -> None:
    if message is None:
        return
    tool_messages = getattr(message, "tool_messages", None)
    if tool_messages:
        state_messages.extend(tool_messages)
        return
    state_messages.append(message)


def build_tau_runtime_policy_payload(
    *,
    initial_answer: str,
    tau_messages: list[Any],
    available_tools: list[str],
    proposed_tool_call: dict[str, Any] | None = None,
    proposed_tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    transcript = []
    user_messages = []
    structured_messages = []
    for message in tau_messages:
        structured_messages.append(_policy_message_dict(message))
        converted = tau_message_to_chat_message(message)
        if converted is None:
            continue
        role = converted["role"]
        content = converted["content"]
        transcript.append(f"{role}: {content}")
        if role == "user" and not content.startswith("Tool result"):
            user_messages.append(content)
    return {
        "task_instruction": "\n".join(user_messages),
        "initial_answer": initial_answer,
        "tool_call": proposed_tool_call,
        "tool_calls": list(proposed_tool_calls or ([] if proposed_tool_call is None else [proposed_tool_call])),
        "available_tools": available_tools,
        "expected_answer": None,
        "rubric": None,
        "metadata": {
            "conversation": "\n".join(transcript),
            "last_user_message": user_messages[-1] if user_messages else "",
            "messages": structured_messages,
        },
    }


def _select_tau_tool_payloads_with_policy(
    *,
    raw: str,
    tau_messages: list[Any],
    available_tools: list[str],
    policies: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tool_payloads = parse_tau_tool_calls(raw)
    if not getattr(policies, "names", []):
        return tool_payloads, []
    policy_payload = build_tau_runtime_policy_payload(
        initial_answer=raw,
        tau_messages=tau_messages,
        available_tools=available_tools,
        proposed_tool_call=tool_payloads[0] if tool_payloads else None,
        proposed_tool_calls=tool_payloads,
    )
    policy_results = policies.evaluate(policy_payload)
    forced_tool = _first_forced_tau_tool(policy_results, available_tools)
    if forced_tool is None:
        return tool_payloads, policy_results
    return [
        {
            "name": forced_tool["tool_name"],
            "arguments": forced_tool.get("tool_input", {}),
        }
    ], policy_results


def _first_forced_tau_tool(
    policy_results: list[dict[str, Any]],
    official_tool_names: list[str],
) -> dict[str, Any] | None:
    official = set(official_tool_names)
    for result in policy_results:
        tool_name = result.get("tool_name")
        tool_input = result.get("tool_input", {})
        if result.get("requires_tool") is True and isinstance(tool_name, str) and isinstance(tool_input, dict):
            if tool_name in official:
                return result
    return None


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
    payloads = _candidate_json_payloads(text)
    for payload in payloads:
        parsed = _parse_tau_tool_payload(payload)
        if parsed:
            return parsed
    return []


def _candidate_json_payloads(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    payloads: list[Any] = []
    if text.startswith("{"):
        try:
            payload, _ = decoder.raw_decode(text)
            payloads.append(payload)
        except json.JSONDecodeError:
            pass
    for match in re.finditer(r"\{", text):
        if match.start() == 0:
            continue
        try:
            payload, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        payloads.append(payload)
    return payloads


def _parse_tau_tool_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    try:
        json.dumps(payload)
    except (TypeError, ValueError):
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


def build_tau_failure_digest(traces: list[dict[str, Any]], *, max_traces: int = 5) -> dict[str, Any]:
    """Compact tau-bench traces into teacher-visible failure evidence."""

    failures = [_summarize_tau_trace(trace) for trace in traces[:max_traces]]
    mode_counts: Counter[str] = Counter()
    tool_call_counts: Counter[str] = Counter()
    rewards: list[float] = []
    for failure in failures:
        mode_counts.update(failure["failure_modes"])
        for tool_name in failure["tool_call_names"]:
            tool_call_counts[tool_name] += 1
        reward = failure.get("reward")
        if isinstance(reward, (int, float)):
            rewards.append(float(reward))
    return {
        "schema": "harnessforge.tau_bench_failure_digest.v1",
        "num_traces": len(traces),
        "summarized_traces": len(failures),
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
        "failure_mode_counts": dict(sorted(mode_counts.items())),
        "tool_call_counts": dict(sorted(tool_call_counts.items())),
        "traces": failures,
        "teacher_guidance": {
            "role": "Use this as evidence about weak-model behavior in the real tau-bench environment.",
            "do_not": [
                "Do not hard-code task ids, reservation ids, user ids, or final answers.",
                "Do not use official tau-bench test split traces for diagnosis or repair.",
                "Do not invent executable helper tools for tau-bench until the adapter can execute them.",
            ],
            "allowed_first_adapter_patches": [
                "prompt_guideline",
                "skill",
                "state_representation",
                "runtime_policy that triggers official tau-bench tools only",
                "validator over trajectory or final answer metadata",
            ],
        },
    }


def build_tau_teacher_context(traces: list[dict[str, Any]], *, max_traces: int = 5) -> dict[str, Any]:
    """Create benchmark_context for teacher diagnosis from tau-bench traces."""

    digest = build_tau_failure_digest(traces, max_traces=max_traces)
    domains = sorted({str(trace.get("domain")) for trace in traces if trace.get("domain") is not None})
    splits = sorted({str(trace.get("split")) for trace in traces if trace.get("split") is not None})
    return {
        "benchmark": "tau-bench text-mode",
        "source_domains": domains,
        "source_splits": splits,
        "split_policy": {
            "teacher_visible": "official train traces only",
            "blind_final": "official test split after the harness is frozen",
        },
        "tau_bench_failure_digest": digest,
        "tau_runtime_evidence": build_tau_runtime_evidence(traces, max_traces=max_traces),
    }


def build_tau_runtime_evidence(
    traces: list[dict[str, Any]],
    *,
    max_traces: int = 5,
    max_windows_per_trace: int = 4,
    window_radius: int = 2,
) -> dict[str, Any]:
    """Expose compact real tau message windows for teacher architecture decisions."""

    windows: list[dict[str, Any]] = []
    for trace in traces[:max_traces]:
        messages = _trace_messages(trace)
        centers = _tau_evidence_centers(messages)
        if not centers and messages:
            centers = [{"index": len(messages) - 1, "reason": "trace_tail"}]
        for center in centers[:max_windows_per_trace]:
            idx = center["index"]
            start = max(0, idx - window_radius)
            end = min(len(messages), idx + window_radius + 1)
            windows.append(
                {
                    "task_id": trace.get("task_id"),
                    "domain": trace.get("domain"),
                    "split": trace.get("split"),
                    "termination_reason": trace.get("termination_reason"),
                    "reward": _trace_reward(trace),
                    "center_index": idx,
                    "window_reason": center["reason"],
                    "messages": [_compact_tau_message(message) for message in messages[start:end]],
                }
            )
    return {
        "schema": "harnessforge.tau_runtime_evidence.v1",
        "trace_windows": windows,
        "message_shape_notes": [
            "Runtime policies receive these entries under context.metadata.messages.",
            "Tau tool-result messages have role=tool and JSON content, but usually do not carry the tool name.",
            "Associate a tool result with the preceding assistant tool_calls entry when tool identity is needed.",
            "Runtime policy decisions appear under assistant raw_data.runtime_policy_results.",
        ],
    }


def _trace_messages(trace: dict[str, Any]) -> list[dict[str, Any]]:
    messages = trace.get("messages")
    if not isinstance(messages, list):
        raw = trace.get("raw")
        messages = raw.get("messages") if isinstance(raw, dict) else []
    return [message for message in messages if isinstance(message, dict)]


def _tau_evidence_centers(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    centers: list[dict[str, Any]] = []
    seen: set[int] = set()
    for idx, message in enumerate(messages):
        reason = _tau_evidence_reason(message)
        if reason and idx not in seen:
            centers.append({"index": idx, "reason": reason})
            seen.add(idx)
    return centers


def _tau_evidence_reason(message: dict[str, Any]) -> str | None:
    raw_data = message.get("raw_data")
    runtime_results = raw_data.get("runtime_policy_results") if isinstance(raw_data, dict) else None
    if isinstance(runtime_results, list):
        if any(isinstance(result, dict) and result.get("requires_tool") is True for result in runtime_results):
            return "runtime_policy_forced_tool"
        if runtime_results:
            return "runtime_policy_evaluated"
    if message.get("tool_calls"):
        return "assistant_tool_call"
    if message.get("role") == "tool" and message.get("error") is True:
        return "tool_error"
    if message.get("role") == "tool":
        return "tool_result"
    return None


def _compact_tau_message(message: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ["role", "turn_idx", "requestor", "error"]:
        if message.get(key) is not None:
            compact[key] = message[key]
    if message.get("content") is not None:
        compact["content"] = _truncate_text(str(message["content"]), 2000)
    if message.get("tool_calls") is not None:
        compact["tool_calls"] = message["tool_calls"]
    raw_data = message.get("raw_data")
    if isinstance(raw_data, dict):
        raw_compact: dict[str, Any] = {}
        if raw_data.get("runtime_policy_results") is not None:
            raw_compact["runtime_policy_results"] = raw_data["runtime_policy_results"]
        if raw_data.get("harnessforge_raw") is not None:
            raw_compact["harnessforge_raw"] = _truncate_text(str(raw_data["harnessforge_raw"]), 1000)
        if raw_compact:
            compact["raw_data"] = raw_compact
    return compact


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _summarize_tau_trace(trace: dict[str, Any]) -> dict[str, Any]:
    messages = _trace_messages(trace)
    assistant_texts = [_message_content(message) for message in messages if _message_role(message) == "assistant"]
    user_texts = [_message_content(message) for message in messages if _message_role(message) == "user"]
    tool_call_names = _trace_tool_call_names(messages)
    repeated = [
        {"text": text, "count": count}
        for text, count in Counter(text for text in assistant_texts if text).most_common(5)
        if count > 1
    ]
    user_identifiers = _extract_user_identifiers(" ".join(user_texts))
    failure_modes = _infer_tau_failure_modes(
        trace=trace,
        tool_call_names=tool_call_names,
        repeated_assistant_texts=repeated,
        user_identifiers=user_identifiers,
    )
    return {
        "task_id": trace.get("task_id"),
        "domain": trace.get("domain"),
        "split": trace.get("split"),
        "termination_reason": trace.get("termination_reason"),
        "reward": _trace_reward(trace),
        "message_count": len(messages),
        "assistant_message_count": len(assistant_texts),
        "user_message_count": len(user_texts),
        "tool_call_count": len(tool_call_names),
        "tool_call_names": tool_call_names,
        "failure_modes": failure_modes,
        "repeated_assistant_messages": repeated,
        "observed_user_identifiers": user_identifiers[:8],
        "first_user_message": user_texts[0] if user_texts else None,
        "last_user_message": user_texts[-1] if user_texts else None,
        "last_assistant_message": assistant_texts[-1] if assistant_texts else None,
    }


def _infer_tau_failure_modes(
    *,
    trace: dict[str, Any],
    tool_call_names: list[str],
    repeated_assistant_texts: list[dict[str, Any]],
    user_identifiers: list[str],
) -> list[str]:
    modes: list[str] = []
    reward = _trace_reward(trace)
    if reward is not None and reward < 1.0:
        modes.append("low_reward")
    if trace.get("termination_reason") == "max_steps":
        modes.append("max_steps")
    if not tool_call_names:
        modes.append("no_tool_calls")
    if repeated_assistant_texts:
        modes.append("repeated_assistant_text")
    if user_identifiers and not tool_call_names:
        modes.append("user_identifiers_not_activated_into_tools")
    return modes


def _trace_reward(trace: dict[str, Any]) -> float | None:
    reward_info = trace.get("reward_info")
    if not isinstance(reward_info, dict):
        return None
    reward = reward_info.get("reward")
    return float(reward) if isinstance(reward, (int, float)) else None


def _trace_tool_call_names(messages: list[Any]) -> list[str]:
    names: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
            if isinstance(name, str):
                names.append(name)
    return names


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        role = message.get("role")
    else:
        role = getattr(message, "role", None)
    return str(role) if role is not None else None


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    return str(content or "").replace("\n", " ").strip()


def _extract_user_identifiers(text: str) -> list[str]:
    patterns = [
        r"\b[A-Z0-9]{5,8}\b",
        r"\b[A-Za-z][A-Za-z0-9_]{2,}[0-9][A-Za-z0-9_]*\b",
    ]
    seen: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if match not in seen:
                seen.append(match)
    return seen


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
            "raw_data": getattr(message, "raw_data", None),
        }
    return {
        "role": data.get("role"),
        "content": data.get("content"),
        "tool_calls": _normalize_tool_calls(data.get("tool_calls")),
        "requestor": data.get("requestor"),
        "error": data.get("error"),
        "turn_idx": data.get("turn_idx"),
        "timestamp": data.get("timestamp"),
        "raw_data": data.get("raw_data"),
    }


def _policy_message_dict(message: Any) -> dict[str, Any]:
    plain = message_to_plain_dict(message)
    return {
        key: value
        for key, value in plain.items()
        if key in {"role", "content", "tool_calls", "requestor", "error", "turn_idx", "timestamp"} and value is not None
    }


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]] | None:
    if tool_calls is None or not isinstance(tool_calls, list):
        return None
    return [_tool_call_to_dict(call) for call in tool_calls]


def _load_tau_tasks(
    tau: dict[str, Any],
    *,
    domain: str,
    split: str,
    task_set_name: str | None,
    num_tasks: int,
    task_ids: list[str] | None = None,
) -> list[Any]:
    registry = tau["registry"]
    loader_name = task_set_name or domain
    loader = registry.get_tasks_loader(loader_name)
    tasks = list(loader(split))
    if task_ids:
        requested = [str(task_id) for task_id in task_ids]
        tasks_by_id = {str(task.id): task for task in tasks}
        missing = [task_id for task_id in requested if task_id not in tasks_by_id]
        if missing:
            raise RuntimeError(f"Unknown tau-bench task ids for task_set={loader_name} split={split}: {missing}")
        return [tasks_by_id[task_id] for task_id in requested]
    if not tasks:
        raise RuntimeError(f"No tau-bench tasks loaded for task_set={loader_name} split={split}")
    return tasks[:num_tasks]


def _complete_sync(client: ChatClient, messages: list[dict[str, str]], *, temperature: float = 0.1) -> str:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return str(asyncio.run(client.complete(messages, temperature=temperature)) or "")
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
