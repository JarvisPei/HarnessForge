from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from agentdistill.models import ModelSettings
from agentdistill.tau_architect_probe import (
    _http_error_details,
    _model_settings_with_overrides,
    build_tau_architect_context,
    load_tau_trace_files,
    run_tau_architect_probe,
)


def _trace(task_id: str = "39") -> dict:
    return {
        "schema": "harnessforge.tau_bench_trace.v1",
        "domain": "airline",
        "split": "train",
        "task_id": task_id,
        "termination_reason": "timeout",
        "reward_info": {
            "reward": 0.0,
            "action_checks": [
                {
                    "action": {
                        "action_id": f"{task_id}_0",
                        "name": "get_user_details",
                        "arguments": {"user_id": "u1"},
                    },
                    "action_match": True,
                    "action_reward": 1.0,
                    "tool_type": "read",
                },
                {
                    "action": {
                        "action_id": f"{task_id}_1",
                        "name": "cancel_reservation",
                        "arguments": {"reservation_id": "R1"},
                    },
                    "action_match": False,
                    "action_reward": 0.0,
                    "tool_type": "write",
                },
            ],
        },
        "raw": {
            "policy": (
                "# Airline Agent Policy\n\n"
                "The current time is 2024-05-15 15:00:00 EST.\n\n"
                "Before taking database actions, obtain explicit confirmation.\n\n"
                "## Cancel flight\n\n"
                "First, obtain the user id and reservation id. If the user does not know reservation ids, "
                "help locate them using available tools. If any portion has already been flown, transfer is needed. "
                "Otherwise, a flight can be cancelled when policy conditions are met.\n\n"
                "## Book flight\n\n"
                "Booking instructions omitted."
            )
        },
        "messages": [
            {"role": "user", "content": "Cancel every upcoming flight.", "turn_idx": 1},
            {
                "role": "assistant",
                "tool_calls": [{"name": "get_user_details", "arguments": {"user_id": "u1"}}],
                "turn_idx": 2,
                "raw_data": {
                    "runtime_policy_phase": "pre_weak",
                    "runtime_policy_results": [
                        {
                            "policy": "candidate_state",
                            "requires_tool": True,
                            "tool_name": "get_user_details",
                            "tool_input": {"user_id": "u1"},
                        }
                    ],
                },
            },
            {"role": "tool", "content": '{"reservations":["R1"]}', "turn_idx": 3},
            {
                "role": "assistant",
                "content": "I will check your remaining reservations.",
                "turn_idx": 4,
                "raw_data": {
                    "runtime_policy_phase": "post_weak",
                    "runtime_policy_results": [{"policy": "candidate_state", "requires_tool": False}],
                },
            },
        ],
    }


def test_load_tau_trace_files_filters_task_ids(tmp_path: Path) -> None:
    (tmp_path / "39.json").write_text(json.dumps(_trace("39")), encoding="utf-8")
    (tmp_path / "41.json").write_text(json.dumps(_trace("41")), encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")

    traces = load_tau_trace_files(tmp_path, task_ids=["41"])

    assert [trace["task_id"] for trace in traces] == ["41"]


def test_build_tau_architect_context_minimal_keeps_forced_tool_and_tail() -> None:
    context = build_tau_architect_context([_trace()], context_mode="minimal")

    windows = context["tau_runtime_evidence"]["trace_windows"]
    assert [(window["task_id"], window["window_reason"]) for window in windows] == [
        ("39", "runtime_policy_forced_tool"),
        ("39", "failed_trace_tail"),
    ]
    assert context["repair_intent"]["preferred_capability"].startswith("stateful runtime policy")


def test_build_tau_architect_context_includes_policy_and_failed_actions() -> None:
    context = build_tau_architect_context([_trace()], context_mode="minimal")

    policy_trace = context["tau_domain_policy_evidence"]["traces"][0]
    assert policy_trace["current_time"] == "2024-05-15 15:00:00 EST"
    assert "## Cancel flight" in policy_trace["policy_excerpt"]

    action_summary = context["tau_bench_failure_digest"]["traces"][0]["action_check_summary"]
    assert action_summary["matched_action_counts"] == {"get_user_details": 1}
    assert action_summary["failed_action_counts"] == {"cancel_reservation": 1}
    assert action_summary["failed_actions"][0]["arguments"] == {"reservation_id": "R1"}

    decision_trace = context["tau_action_decision_evidence"]["traces"][0]
    assert decision_trace["observed_tool_argument_shapes"]["cancel_reservation"] == [{"reservation_id": "str"}]
    assert decision_trace["failed_actions"][0]["failed_action"]["name"] == "cancel_reservation"
    assert decision_trace["failed_actions"][0]["related_tool_results"][0]["tool_name"] == "get_user_details"


def test_build_tau_architect_context_decision_mode_drops_runtime_windows() -> None:
    context = build_tau_architect_context([_trace()], context_mode="decision")

    assert context["tau_runtime_evidence"]["trace_windows"] == []
    assert context["tau_action_decision_evidence"]["traces"][0]["failed_actions"]
    assert context["tau_domain_policy_evidence"]["traces"][0]["current_time"] == "2024-05-15 15:00:00 EST"


def test_model_settings_overrides_can_disable_reasoning() -> None:
    settings = ModelSettings(
        role="teacher",
        provider="openai",
        base_url="https://example.test/v1",
        api_key="key",
        model="teacher",
        reasoning_effort="high",
        timeout_seconds=600,
        max_retries=2,
    )

    updated = _model_settings_with_overrides(settings, reasoning_effort="none", timeout=30, max_retries=0)

    assert updated.reasoning_effort is None
    assert updated.timeout_seconds == 30
    assert updated.max_retries == 0


def test_http_error_details_records_status_and_short_body() -> None:
    request = httpx.Request("POST", "https://relay.example.test/v1/chat/completions")
    response = httpx.Response(400, request=request, text='{"error":"bad model"}')
    exc = httpx.HTTPStatusError("bad request", request=request, response=response)

    details = _http_error_details(exc)

    assert details == {"http_status": 400, "response_text": '{"error":"bad model"}'}


def test_tau_architect_probe_dry_run_does_not_require_model_env(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "39.json").write_text(json.dumps(_trace()), encoding="utf-8")
    output_dir = tmp_path / "probe"
    for name in ["TEACHER_BASE_URL", "TEACHER_API_KEY", "TEACHER_MODEL"]:
        monkeypatch.delenv(name, raising=False)

    summary = asyncio.run(
        run_tau_architect_probe(
            repo_root=repo_root,
            trace_dir=trace_dir,
            task_ids=["39"],
            output_dir=output_dir,
            profile=None,
            context_mode="minimal",
            reasoning_effort=None,
            timeout=None,
            max_retries=None,
            temperature=0.1,
            dry_run=True,
        )
    )

    assert summary["status"] == "dry_run"
    assert summary["model"] is None
    assert (output_dir / "teacher_payload.json").exists()
    assert (output_dir / "summary.json").exists()
