from __future__ import annotations

from dataclasses import dataclass

from agentdistill.tau_bench import (
    _append_tau_incoming_message,
    _first_forced_tau_tool,
    _select_tau_tool_payloads_with_policy,
    build_tau_failure_digest,
    build_tau_runtime_policy_payload,
    build_tau_teacher_context,
    message_to_plain_dict,
    parse_tau_tool_calls,
    simulation_run_to_trace,
    task_error_to_trace,
    tau_message_to_chat_message,
)


@dataclass
class FakeToolCall:
    name: str
    arguments: dict
    id: str = "call_1"
    requestor: str = "assistant"


@dataclass
class FakeMessage:
    role: str
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None
    requestor: str | None = None
    error: bool | None = None
    turn_idx: int | None = None
    timestamp: str | None = None
    id: str | None = None


@dataclass
class FakeMultiToolMessage:
    tool_messages: list[FakeMessage]


class FakePolicies:
    def __init__(self, results: list[dict] | None = None):
        self.names = ["guard_policy"]
        self.results = results or []
        self.payloads: list[dict] = []

    def evaluate(self, payload: dict) -> list[dict]:
        self.payloads.append(payload)
        return self.results


def test_parse_tau_tool_call_accepts_arguments() -> None:
    payload = '{"tool_call": {"name": "get_order", "arguments": {"order_id": "O-1"}}}'

    assert parse_tau_tool_calls(payload) == [
        {"name": "get_order", "arguments": {"order_id": "O-1"}}
    ]


def test_parse_tau_tool_call_accepts_input_alias_and_fences() -> None:
    payload = '```json\n{"tool_call": {"name": "lookup", "input": {"x": 1}}}\n```'

    assert parse_tau_tool_calls(payload) == [{"name": "lookup", "arguments": {"x": 1}}]


def test_parse_tau_tool_call_handles_empty_content() -> None:
    assert parse_tau_tool_calls(None) == []
    assert parse_tau_tool_calls("") == []


def test_tau_message_to_chat_message_converts_tool_result_to_user_message() -> None:
    message = FakeMessage(
        role="tool",
        content='{"ok": true}',
        requestor="assistant",
        error=False,
        id="tool_1",
    )

    converted = tau_message_to_chat_message(message)

    assert converted is not None
    assert converted["role"] == "user"
    assert "Tool result" in converted["content"]
    assert "ok" in converted["content"]


def test_append_tau_incoming_message_keeps_user_context() -> None:
    state_messages: list[object] = []
    user_message = FakeMessage(role="user", content="Cancel reservation EHGLP3.")

    _append_tau_incoming_message(state_messages, user_message)
    _append_tau_incoming_message(state_messages, None)

    assert state_messages == [user_message]


def test_append_tau_incoming_message_expands_tool_messages() -> None:
    state_messages: list[object] = []
    tool_message = FakeMessage(role="tool", content='{"ok": true}', id="tool_1")

    _append_tau_incoming_message(state_messages, FakeMultiToolMessage(tool_messages=[tool_message]))

    assert state_messages == [tool_message]


def test_build_tau_runtime_policy_payload_includes_conversation_metadata() -> None:
    payload = build_tau_runtime_policy_payload(
        initial_answer="How can I help you today?",
        tau_messages=[
            FakeMessage(role="user", content="Cancel reservation EHGLP3."),
            FakeMessage(role="assistant", content="How can I help you today?"),
            FakeMessage(role="user", content="My user ID is emma_kim_9957."),
        ],
        available_tools=["get_user_details", "get_reservation_details"],
    )

    assert payload["task_instruction"] == "Cancel reservation EHGLP3.\nMy user ID is emma_kim_9957."
    assert payload["initial_answer"] == "How can I help you today?"
    assert payload["available_tools"] == ["get_user_details", "get_reservation_details"]
    assert payload["metadata"]["last_user_message"] == "My user ID is emma_kim_9957."
    assert "assistant: How can I help you today?" in payload["metadata"]["conversation"]


def test_build_tau_runtime_policy_payload_includes_proposed_tool_call() -> None:
    proposed = {"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}

    payload = build_tau_runtime_policy_payload(
        initial_answer='{"tool_call": {"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}}',
        tau_messages=[FakeMessage(role="user", content="Cancel my matching reservation.")],
        available_tools=["cancel_reservation", "get_reservation_details"],
        proposed_tool_call=proposed,
        proposed_tool_calls=[proposed],
    )

    assert payload["tool_call"] == proposed
    assert payload["tool_calls"] == [proposed]


def test_first_forced_tau_tool_ignores_non_official_tools() -> None:
    forced = _first_forced_tau_tool(
        [
            {"requires_tool": True, "tool_name": "helper_tool", "tool_input": {}},
            {"requires_tool": True, "tool_name": "get_user_details", "tool_input": {"user_id": "u1"}},
        ],
        ["get_user_details"],
    )

    assert forced == {"requires_tool": True, "tool_name": "get_user_details", "tool_input": {"user_id": "u1"}}


def test_select_tau_tool_payloads_with_policy_can_replace_proposed_tool_call() -> None:
    policies = FakePolicies(
        [
            {
                "requires_tool": True,
                "tool_name": "get_reservation_details",
                "tool_input": {"reservation_id": "EHGLP3"},
            }
        ]
    )

    tool_payloads, policy_results = _select_tau_tool_payloads_with_policy(
        raw='{"tool_call": {"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}}',
        tau_messages=[FakeMessage(role="user", content="Cancel the Philadelphia to LaGuardia reservation.")],
        available_tools=["cancel_reservation", "get_reservation_details"],
        policies=policies,
    )

    assert tool_payloads == [{"name": "get_reservation_details", "arguments": {"reservation_id": "EHGLP3"}}]
    assert policy_results == policies.results
    assert policies.payloads[0]["tool_call"] == {"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}


def test_select_tau_tool_payloads_with_policy_ignores_non_official_replacement() -> None:
    policies = FakePolicies(
        [
            {"requires_tool": True, "tool_name": "helper_tool", "tool_input": {"reservation_id": "EHGLP3"}},
        ]
    )

    tool_payloads, policy_results = _select_tau_tool_payloads_with_policy(
        raw='{"tool_call": {"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}}',
        tau_messages=[FakeMessage(role="user", content="Cancel the matching reservation.")],
        available_tools=["cancel_reservation", "get_reservation_details"],
        policies=policies,
    )

    assert tool_payloads == [{"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}]
    assert policy_results == policies.results


def test_select_tau_tool_payloads_with_policy_still_supports_no_tool_fallback() -> None:
    policies = FakePolicies(
        [
            {"requires_tool": True, "tool_name": "get_user_details", "tool_input": {"user_id": "u1"}},
        ]
    )

    tool_payloads, _ = _select_tau_tool_payloads_with_policy(
        raw="I can help with that.",
        tau_messages=[FakeMessage(role="user", content="My user ID is u1.")],
        available_tools=["get_user_details"],
        policies=policies,
    )

    assert tool_payloads == [{"name": "get_user_details", "arguments": {"user_id": "u1"}}]
    assert policies.payloads[0]["tool_call"] is None


def test_message_to_plain_dict_keeps_tool_calls() -> None:
    message = FakeMessage(
        role="assistant",
        tool_calls=[FakeToolCall(name="get_user", arguments={"user_id": "u1"})],
        turn_idx=2,
    )

    plain = message_to_plain_dict(message)

    assert plain["role"] == "assistant"
    assert plain["tool_calls"] == message.tool_calls
    assert plain["turn_idx"] == 2


def test_simulation_run_to_trace_with_plain_dict() -> None:
    run = {
        "id": "sim_1",
        "task_id": "task_1",
        "termination_reason": "user_stop",
        "reward_info": {"reward": 1.0},
        "agent_cost": 0.1,
        "user_cost": 0.2,
        "duration": 3.0,
        "seed": 301,
        "messages": [{"role": "assistant", "content": "Hi"}],
    }

    trace = simulation_run_to_trace(run, domain="airline", split="train")

    assert trace["schema"] == "harnessforge.tau_bench_trace.v1"
    assert trace["domain"] == "airline"
    assert trace["split"] == "train"
    assert trace["task_id"] == "task_1"
    assert trace["reward_info"] == {"reward": 1.0}
    assert trace["messages"] == [
        {
            "role": "assistant",
            "content": "Hi",
            "tool_calls": None,
            "requestor": None,
            "error": None,
            "turn_idx": None,
            "timestamp": None,
        }
    ]


def test_task_error_to_trace_records_adapter_error() -> None:
    task = type("Task", (), {"id": "task_1"})()

    trace = task_error_to_trace(task, domain="airline", split="train", exc=ValueError("bad response"))

    assert trace["task_id"] == "task_1"
    assert trace["termination_reason"] == "adapter_error"
    assert trace["reward_info"] is None
    assert trace["error"] == {"type": "ValueError", "message": "bad response"}


def test_build_tau_failure_digest_identifies_no_tool_loop() -> None:
    trace = {
        "domain": "airline",
        "split": "train",
        "task_id": "0",
        "termination_reason": "max_steps",
        "reward_info": {"reward": 0.0},
        "messages": [
            {"role": "assistant", "content": "How can I help you today?", "tool_calls": None},
            {"role": "user", "content": "Cancel reservation EHGLP3.", "tool_calls": None},
            {"role": "assistant", "content": "How can I help you today?", "tool_calls": None},
            {"role": "user", "content": "My user ID is emma_kim_9957.", "tool_calls": None},
        ],
    }

    digest = build_tau_failure_digest([trace])

    assert digest["schema"] == "harnessforge.tau_bench_failure_digest.v1"
    assert digest["reward_mean"] == 0.0
    assert digest["failure_mode_counts"]["max_steps"] == 1
    assert digest["failure_mode_counts"]["no_tool_calls"] == 1
    assert digest["failure_mode_counts"]["repeated_assistant_text"] == 1
    assert digest["failure_mode_counts"]["user_identifiers_not_activated_into_tools"] == 1
    failure = digest["traces"][0]
    assert failure["observed_user_identifiers"] == ["EHGLP3", "emma_kim_9957"]
    assert failure["repeated_assistant_messages"] == [{"text": "How can I help you today?", "count": 2}]


def test_build_tau_teacher_context_marks_split_boundary() -> None:
    trace = {
        "domain": "retail",
        "split": "train",
        "task_id": "r1",
        "termination_reason": "max_steps",
        "reward_info": {"reward": 0.0},
        "messages": [],
    }

    context = build_tau_teacher_context([trace])

    assert context["benchmark"] == "tau-bench text-mode"
    assert context["source_domains"] == ["retail"]
    assert context["source_splits"] == ["train"]
    assert context["split_policy"]["teacher_visible"] == "official train traces only"
    assert "tau_bench_failure_digest" in context
