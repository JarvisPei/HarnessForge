from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import typer
import httpx
from dotenv import load_dotenv
from rich.console import Console

from agentdistill.diagnosis import parse_diagnosis
from agentdistill.models import ChatClient, ModelSettings, load_model_settings
from agentdistill.prompt_loader import load_teacher_system_prompt
from agentdistill.tau_bench import build_tau_teacher_context
from agentdistill.teacher_prompt import build_teacher_messages


ContextMode = Literal["full", "slim", "minimal", "decision"]

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    trace_dir: Path = typer.Option(..., "--trace-dir", help="Directory containing tau trace JSON files."),
    task_ids: list[str] | None = typer.Option(None, "--task-id", help="Task id to include; repeat for multiple."),
    output_dir: Path = typer.Option(
        Path("outputs/tau_bench_teacher_probe/tau_architect_probe"),
        "--output-dir",
        "-o",
    ),
    profile: str | None = typer.Option(None, "--profile", "-p"),
    context_mode: ContextMode = typer.Option("slim", "--context-mode"),
    reasoning_effort: str | None = typer.Option(None, "--reasoning-effort"),
    timeout: float | None = typer.Option(None, "--timeout"),
    max_retries: int | None = typer.Option(None, "--max-retries", min=0),
    temperature: float = typer.Option(0.1, "--temperature"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only write the teacher payload; do not call the model."),
) -> None:
    """Ask the teacher to architect a tau-bench harness patch from saved traces."""

    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=repo_root / ".env", override=False)
    output_dir = output_dir if output_dir.is_absolute() else repo_root / output_dir
    trace_dir = trace_dir if trace_dir.is_absolute() else repo_root / trace_dir
    try:
        summary = asyncio.run(
            run_tau_architect_probe(
                repo_root=repo_root,
                trace_dir=trace_dir,
                task_ids=task_ids,
                output_dir=output_dir,
                profile=profile,
                context_mode=context_mode,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                max_retries=max_retries,
                temperature=temperature,
                dry_run=dry_run,
            )
        )
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(json.dumps(summary, indent=2, ensure_ascii=False))


async def run_tau_architect_probe(
    *,
    repo_root: Path,
    trace_dir: Path,
    task_ids: list[str] | None,
    output_dir: Path,
    profile: str | None,
    context_mode: ContextMode,
    reasoning_effort: str | None,
    timeout: float | None,
    max_retries: int | None,
    temperature: float,
    dry_run: bool,
) -> dict[str, Any]:
    traces = load_tau_trace_files(trace_dir, task_ids=task_ids)
    if not traces:
        raise RuntimeError(f"No tau traces found in {trace_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    context = build_tau_architect_context(traces, context_mode=context_mode)
    attach_active_harness_evidence(context, repo_root=repo_root)
    messages = build_tau_architect_messages(repo_root, context=context, context_mode=context_mode)
    payload = json.loads(messages[1]["content"])
    (output_dir / "teacher_payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    run_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trace_dir": str(trace_dir),
        "task_ids": [str(trace.get("task_id")) for trace in traces],
        "output_dir": str(output_dir),
        "profile": profile,
        "context_mode": context_mode,
        "dry_run": dry_run,
        "model": None,
        "provider": None,
        "reasoning_effort": None,
        "timeout_seconds": None,
        "max_retries": None,
        "payload_bytes": len((output_dir / "teacher_payload.json").read_bytes()),
    }
    if dry_run:
        summary = {**run_config, "status": "dry_run"}
        (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    settings = _model_settings_with_overrides(
        load_model_settings("teacher", profile),
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        max_retries=max_retries,
    )
    run_config.update(
        {
            "model": settings.model,
            "provider": settings.provider,
            "reasoning_effort": settings.reasoning_effort,
            "timeout_seconds": settings.timeout_seconds,
            "max_retries": settings.max_retries,
        }
    )
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        raw = await ChatClient(settings).complete(messages, temperature=temperature)
    except Exception as exc:
        error = {
            **run_config,
            "status": "model_error",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            **_http_error_details(exc),
        }
        (output_dir / "error.json").write_text(json.dumps(error, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError(f"Teacher probe failed; wrote {output_dir / 'error.json'}") from exc

    (output_dir / "patch_raw.json").write_text(raw, encoding="utf-8")
    diagnosis = parse_diagnosis(raw)
    parsed = diagnosis.model_dump()
    (output_dir / "patch_parsed.json").write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        **run_config,
        "status": "completed",
        "parse_status": diagnosis.parse_status,
        "patch_type": diagnosis.patch_type,
        "failure_categories": diagnosis.failure_categories,
        "num_patch_bundles": len(diagnosis.patch_bundles),
        "diagnosis": diagnosis.diagnosis[:1000],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def load_tau_trace_files(trace_dir: Path, *, task_ids: list[str] | None = None) -> list[dict[str, Any]]:
    wanted = {str(task_id) for task_id in task_ids or []}
    traces: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        if wanted and path.stem not in wanted:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            traces.append(data)
    return traces


def build_tau_architect_context(traces: list[dict[str, Any]], *, context_mode: ContextMode) -> dict[str, Any]:
    context = build_tau_teacher_context(traces, max_traces=len(traces))
    if context_mode == "decision":
        evidence = context.get("tau_runtime_evidence")
        if isinstance(evidence, dict):
            evidence["trace_windows"] = []
    elif context_mode != "full":
        evidence = context.get("tau_runtime_evidence")
        if isinstance(evidence, dict):
            evidence["trace_windows"] = _select_tau_architect_windows(
                evidence.get("trace_windows", []),
                context_mode=context_mode,
            )
    context["repair_intent"] = {
        "goal": "Architect a general harness change for weak-model long-horizon tau-bench progress failures.",
        "preferred_capability": (
            "stateful runtime policy progress controller over official tau-bench tools when justified by trace evidence"
        ),
        "do_not": [
            "do not hard-code task ids, user ids, reservation ids, or final answers",
            "do not write a one-task guard unless the evidence only supports that",
        ],
    }
    return context


def attach_active_harness_evidence(
    context: dict[str, Any],
    *,
    repo_root: Path,
    max_files: int = 1,
    max_chars_per_file: int = 7000,
    max_total_chars: int = 7000,
) -> None:
    """Attach active generated harness files that may affect the saved trace."""

    policy_names = _runtime_policy_names_from_context(context)
    files: list[dict[str, Any]] = []
    total_chars = 0
    for policy_name in policy_names:
        path = repo_root / "harness" / "runtime_policies" / f"{policy_name}.py"
        if not path.exists() or not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        remaining = max_total_chars - total_chars
        if remaining <= 0:
            break
        limit = min(max_chars_per_file, remaining)
        files.append(
            {
                "path": str(path.relative_to(repo_root)),
                "type": "runtime_policy",
                "content": _truncate_text(content, limit),
            }
        )
        total_chars += len(files[-1]["content"])
        if len(files) >= max_files:
            break
    if files:
        context["active_harness_evidence"] = {
            "schema": "harnessforge.active_harness_evidence.v1",
            "files": files,
            "notes": [
                "These are active generated harness files present in the experiment workspace.",
                "Use them to extend or replace existing behavior instead of blindly duplicating policies.",
            ],
        }


def _runtime_policy_names_from_context(context: dict[str, Any]) -> list[str]:
    digest = context.get("tau_bench_failure_digest")
    counts = digest.get("runtime_policy_counts") if isinstance(digest, dict) else None
    if not isinstance(counts, dict):
        return []
    names = [name for name in counts if isinstance(name, str)]
    return sorted(names, key=lambda name: ("progress" not in name and "cancel" not in name, name))


def _select_tau_architect_windows(windows: Any, *, context_mode: ContextMode) -> list[dict[str, Any]]:
    if not isinstance(windows, list):
        return []
    if context_mode == "minimal":
        priority = ["runtime_policy_forced_tool", "failed_trace_tail"]
        limit_per_trace = 2
    else:
        priority = ["runtime_policy_forced_tool", "tool_result", "assistant_tool_call", "failed_trace_tail"]
        limit_per_trace = 4
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    trace_ids = []
    for window in windows:
        if isinstance(window, dict):
            task_id = str(window.get("task_id"))
            if task_id not in trace_ids:
                trace_ids.append(task_id)
    for task_id in trace_ids:
        per_trace = [window for window in windows if isinstance(window, dict) and str(window.get("task_id")) == task_id]
        chosen_for_trace = 0
        for reason in priority:
            for window in per_trace:
                key = (task_id, str(window.get("window_reason")))
                if window.get("window_reason") == reason and key not in seen:
                    selected.append(window)
                    seen.add(key)
                    chosen_for_trace += 1
                    break
            if chosen_for_trace >= limit_per_trace:
                break
    return selected


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_tau_architect_messages(
    repo_root: Path,
    *,
    context: dict[str, Any],
    context_mode: ContextMode,
) -> list[dict[str, str]]:
    return build_teacher_messages(
        load_teacher_system_prompt(repo_root),
        task_id=f"tau_architect_probe_{context_mode}",
        task_instruction=(
            "Use the official train traces in benchmark_context to decide whether a harness patch is justified for "
            "long-horizon tau-bench agent progress failures."
        ),
        expected_answer=None,
        rubric=(
            "Prefer context_request if insufficient. If patching, use a runtime_policy progress controller that "
            "reconstructs state from metadata.messages and only forces official tau-bench tools."
        ),
        weak_system_prompt=(repo_root / "prompts" / "weak_system.md").read_text(encoding="utf-8"),
        weak_answer="See benchmark_context.",
        initial_weak_answer="See benchmark_context.",
        tool_call=None,
        tool_result=None,
        runtime_policy_results=[],
        benchmark_context=context,
    )


def _model_settings_with_overrides(
    settings: ModelSettings,
    *,
    reasoning_effort: str | None,
    timeout: float | None,
    max_retries: int | None,
) -> ModelSettings:
    return replace(
        settings,
        reasoning_effort=_normalize_optional_override(reasoning_effort, settings.reasoning_effort),
        timeout_seconds=timeout if timeout is not None else settings.timeout_seconds,
        max_retries=max_retries if max_retries is not None else settings.max_retries,
    )


def _normalize_optional_override(value: str | None, default: str | None) -> str | None:
    if value is None:
        return default
    value = value.strip()
    if value.lower() in {"", "none", "off", "null"}:
        return None
    return value


def _http_error_details(exc: Exception) -> dict[str, Any]:
    if not isinstance(exc, httpx.HTTPStatusError):
        return {}
    text = " ".join(exc.response.text.strip().split())
    if len(text) > 500:
        text = text[:497] + "..."
    return {
        "http_status": exc.response.status_code,
        "response_text": text,
    }


if __name__ == "__main__":
    app()
