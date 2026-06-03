from __future__ import annotations

from dataclasses import dataclass

from agentdistill.tau_bench import (
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
