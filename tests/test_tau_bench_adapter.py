from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentdistill.models import ModelSettings
from agentdistill.tau_bench import (
    TauHarnessSettings,
    _append_tau_incoming_message,
    _first_forced_tau_tool,
    _first_tau_tool_denial,
    _load_tau_tasks,
    _make_tau_user_llm_completion_shim,
    _register_harnessforge_agent,
    _resolve_optional_repo_path,
    _select_pre_weak_tau_tool_payloads,
    _select_tau_tool_payloads_with_policy,
    build_tau_failure_digest,
    build_tau_runtime_policy_payload,
    build_tau_runtime_evidence,
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


@dataclass
class FakeTask:
    id: str


class FakeTauRegistry:
    def __init__(self, tasks: list[FakeTask]):
        self.tasks = tasks

    def get_tasks_loader(self, name: str):
        assert name == "airline"
        return lambda split: self.tasks


class FakePolicies:
    def __init__(self, results: list[dict] | None = None):
        self.names = ["guard_policy"]
        self.results = results or []
        self.payloads: list[dict] = []

    def evaluate(self, payload: dict) -> list[dict]:
        self.payloads.append(payload)
        return self.results


class FakeAgentRegistry:
    def __init__(self):
        self.factories: dict[str, object] = {}

    def get_agent_factory(self, name: str):
        return self.factories.get(name)

    def register_agent_factory(self, factory, name: str) -> None:
        self.factories[name] = factory


class FakeHalfDuplexAgent:
    def __init__(self, tools, domain_policy):
        self.tools = tools
        self.domain_policy = domain_policy


class FakeAssistantMessage:
    def __init__(self, role="assistant", content=None, tool_calls=None, raw_data=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.raw_data = raw_data

    @classmethod
    def text(cls, content: str, raw_data=None):
        return cls(role="assistant", content=content, tool_calls=None, raw_data=raw_data)


@dataclass
class FakeOfficialTool:
    name: str
    openai_schema: dict | None = None


def _make_tau_agent(tmp_path: Path, monkeypatch, policies: FakePolicies):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "weak_system.md").write_text("Weak system.", encoding="utf-8")
    for subdir in ["skills", "guidelines", "validators", "runtime_policies"]:
        (tmp_path / "harness" / subdir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "agentdistill.tau_bench.load_model_settings",
        lambda role, profile=None: ModelSettings(
            role="weak",
            provider="openai",
            base_url="https://example.com/v1",
            api_key="k",
            model="weak",
        ),
    )
    monkeypatch.setattr("agentdistill.tau_bench.RuntimePolicyRegistry", lambda path: policies)

    registry = FakeAgentRegistry()
    tau = {
        "registry": registry,
        "HalfDuplexAgent": FakeHalfDuplexAgent,
        "AssistantMessage": FakeAssistantMessage,
        "ToolCall": FakeToolCall,
        "is_valid_agent_history_message": lambda message: True,
    }
    _register_harnessforge_agent(tau, "fake_agent", TauHarnessSettings(repo_root=tmp_path))
    factory = registry.factories["fake_agent"]
    return factory(tools=[FakeOfficialTool("get_user_details")], domain_policy="Domain policy.")


def test_resolve_optional_repo_path_keeps_none() -> None:
    assert _resolve_optional_repo_path(Path("/repo"), None) is None


def test_resolve_optional_repo_path_resolves_relative_to_repo(tmp_path: Path) -> None:
    assert _resolve_optional_repo_path(tmp_path, Path("outputs/policies")) == (tmp_path / "outputs/policies").resolve()


def test_resolve_optional_repo_path_preserves_absolute(tmp_path: Path) -> None:
    absolute = (tmp_path / "policies").resolve()
    assert _resolve_optional_repo_path(Path("/repo"), absolute) == absolute


def test_parse_tau_tool_call_accepts_arguments() -> None:
    payload = '{"tool_call": {"name": "get_order", "arguments": {"order_id": "O-1"}}}'

    assert parse_tau_tool_calls(payload) == [
        {"name": "get_order", "arguments": {"order_id": "O-1"}}
    ]


def test_parse_tau_tool_call_accepts_input_alias_and_fences() -> None:
    payload = '```json\n{"tool_call": {"name": "lookup", "input": {"x": 1}}}\n```'

    assert parse_tau_tool_calls(payload) == [{"name": "lookup", "arguments": {"x": 1}}]


def test_parse_tau_tool_call_extracts_embedded_tool_json() -> None:
    payload = 'I am checking now.{"tool_call": {"name": "get_user_details", "arguments": {"user_id": "u1"}}}'

    assert parse_tau_tool_calls(payload) == [{"name": "get_user_details", "arguments": {"user_id": "u1"}}]


def test_parse_tau_tool_call_ignores_embedded_non_tool_json() -> None:
    payload = 'The profile is {"user_id": "u1"} and no tool is needed.'

    assert parse_tau_tool_calls(payload) == []


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


def test_tau_user_llm_completion_shim_uses_chat_client(monkeypatch) -> None:
    captured = {}

    async def fake_complete(self, messages, temperature=0.2):
        captured["settings"] = self.settings
        captured["messages"] = messages
        captured["temperature"] = temperature
        return "hello from shim"

    def original_completion(**kwargs):
        raise AssertionError("original LiteLLM completion should not be used for text-only shim calls")

    monkeypatch.setenv("TAU_USER_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("TAU_USER_API_KEY", "k")
    monkeypatch.setattr("agentdistill.tau_bench.ChatClient.complete", fake_complete)

    shim = _make_tau_user_llm_completion_shim(original_completion)
    response = shim(model="openai/gpt-5.5", messages=[{"role": "user", "content": "hi"}], temperature=0.3)

    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "hello from shim"
    assert response.to_dict()["choices"][0]["message"]["content"] == "hello from shim"
    assert captured["settings"].model == "gpt-5.5"
    assert captured["settings"].base_url == "https://example.com/v1"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["temperature"] == 0.3


def test_tau_user_llm_completion_shim_falls_back_for_tool_calls() -> None:
    calls = {}

    def original_completion(**kwargs):
        calls["kwargs"] = kwargs
        return {"ok": True}

    shim = _make_tau_user_llm_completion_shim(original_completion)
    response = shim(model="gpt-5.5", messages=[{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

    assert response == {"ok": True}
    assert calls["kwargs"]["tools"] == [{"type": "function"}]


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
            FakeMessage(role="assistant", tool_calls=[FakeToolCall(name="get_user_details", arguments={"user_id": "u1"})]),
            FakeMessage(role="tool", content='{"reservations": ["EHGLP3"]}', id="tool_1"),
            FakeMessage(role="user", content="My user ID is emma_kim_9957."),
        ],
        available_tools=["get_user_details", "get_reservation_details"],
    )

    assert payload["task_instruction"] == "Cancel reservation EHGLP3.\nMy user ID is emma_kim_9957."
    assert payload["initial_answer"] == "How can I help you today?"
    assert payload["available_tools"] == ["get_user_details", "get_reservation_details"]
    assert payload["metadata"]["last_user_message"] == "My user ID is emma_kim_9957."
    assert "assistant: Assistant tool call" in payload["metadata"]["conversation"]
    assert payload["metadata"]["messages"] == [
        {"role": "user", "content": "Cancel reservation EHGLP3."},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "name": "get_user_details", "arguments": {"user_id": "u1"}, "requestor": "assistant"}
            ],
        },
        {"role": "tool", "content": '{"reservations": ["EHGLP3"]}'},
        {"role": "user", "content": "My user ID is emma_kim_9957."},
    ]


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


def test_first_tau_tool_denial_requires_matching_proposed_tool_and_response() -> None:
    denial = _first_tau_tool_denial(
        [
            {"deny_tool": True, "tool_name": "get_user_details", "assistant_response": "wrong tool"},
            {"deny_tool": True, "tool_name": "cancel_reservation", "assistant_response": "I cannot cancel this reservation."},
        ],
        [{"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}],
    )

    assert denial == {
        "deny_tool": True,
        "tool_name": "cancel_reservation",
        "tool_input": {"reservation_id": "MZDDS4"},
        "assistant_response": "I cannot cancel this reservation.",
        "reason": "runtime policy denied proposed tool call",
    }


def test_select_tau_tool_payloads_with_policy_can_deny_proposed_tool_call() -> None:
    policies = FakePolicies(
        [
            {
                "deny_tool": True,
                "tool_name": "cancel_reservation",
                "assistant_response": "I cannot cancel this reservation because it is not refundable under policy.",
                "reason": "cancellation eligibility not met",
            }
        ]
    )

    tool_payloads, policy_results, denial = _select_tau_tool_payloads_with_policy(
        raw='{"tool_call": {"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}}',
        tau_messages=[FakeMessage(role="user", content="Cancel the Philadelphia to LaGuardia reservation.")],
        available_tools=["cancel_reservation", "get_reservation_details"],
        policies=policies,
    )

    assert tool_payloads == []
    assert policy_results == policies.results
    assert denial == {
        "deny_tool": True,
        "tool_name": "cancel_reservation",
        "tool_input": {"reservation_id": "MZDDS4"},
        "assistant_response": "I cannot cancel this reservation because it is not refundable under policy.",
        "reason": "cancellation eligibility not met",
    }
    assert policies.payloads[0]["runtime_policy_phase"] == "post_weak"
    assert policies.payloads[0]["tool_call"] == {"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}


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

    tool_payloads, policy_results, denial = _select_tau_tool_payloads_with_policy(
        raw='{"tool_call": {"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}}',
        tau_messages=[FakeMessage(role="user", content="Cancel the Philadelphia to LaGuardia reservation.")],
        available_tools=["cancel_reservation", "get_reservation_details"],
        policies=policies,
    )

    assert tool_payloads == [{"name": "get_reservation_details", "arguments": {"reservation_id": "EHGLP3"}}]
    assert policy_results == policies.results
    assert denial is None
    assert policies.payloads[0]["tool_call"] == {"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}


def test_select_tau_tool_payloads_with_policy_ignores_non_official_replacement() -> None:
    policies = FakePolicies(
        [
            {"requires_tool": True, "tool_name": "helper_tool", "tool_input": {"reservation_id": "EHGLP3"}},
        ]
    )

    tool_payloads, policy_results, denial = _select_tau_tool_payloads_with_policy(
        raw='{"tool_call": {"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}}',
        tau_messages=[FakeMessage(role="user", content="Cancel the matching reservation.")],
        available_tools=["cancel_reservation", "get_reservation_details"],
        policies=policies,
    )

    assert tool_payloads == [{"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}]
    assert policy_results == policies.results
    assert denial is None


def test_select_tau_tool_payloads_with_policy_still_supports_no_tool_fallback() -> None:
    policies = FakePolicies(
        [
            {"requires_tool": True, "tool_name": "get_user_details", "tool_input": {"user_id": "u1"}},
        ]
    )

    tool_payloads, _, denial = _select_tau_tool_payloads_with_policy(
        raw="I can help with that.",
        tau_messages=[FakeMessage(role="user", content="My user ID is u1.")],
        available_tools=["get_user_details"],
        policies=policies,
    )

    assert tool_payloads == [{"name": "get_user_details", "arguments": {"user_id": "u1"}}]
    assert denial is None
    assert policies.payloads[0]["tool_call"] is None


def test_select_pre_weak_tau_tool_payloads_can_force_without_weak_raw() -> None:
    policies = FakePolicies(
        [
            {"requires_tool": True, "tool_name": "get_user_details", "tool_input": {"user_id": "u1"}},
        ]
    )

    tool_payloads, policy_results = _select_pre_weak_tau_tool_payloads(
        tau_messages=[FakeMessage(role="user", content="My user ID is u1.")],
        available_tools=["get_user_details"],
        policies=policies,
    )

    assert tool_payloads == [{"name": "get_user_details", "arguments": {"user_id": "u1"}}]
    assert policy_results == policies.results
    assert policies.payloads[0]["initial_answer"] == ""
    assert policies.payloads[0]["tool_call"] is None
    assert policies.payloads[0]["runtime_policy_phase"] == "pre_weak"


def test_tau_agent_pre_weak_policy_skips_weak_call(tmp_path: Path, monkeypatch) -> None:
    policies = FakePolicies(
        [
            {"requires_tool": True, "tool_name": "get_user_details", "tool_input": {"user_id": "u1"}},
        ]
    )
    agent = _make_tau_agent(tmp_path, monkeypatch, policies)

    def fail_complete_sync(*args, **kwargs):
        raise AssertionError("weak model should not be called when pre-weak policy forces a tool")

    monkeypatch.setattr("agentdistill.tau_bench._complete_sync", fail_complete_sync)
    state = agent.get_init_state()

    assistant, state = agent.generate_next_message(FakeMessage(role="user", content="My user ID is u1."), state)

    assert assistant.content is None
    assert assistant.tool_calls == [FakeToolCall(id=assistant.tool_calls[0].id, name="get_user_details", arguments={"user_id": "u1"})]
    assert assistant.raw_data["runtime_policy_phase"] == "pre_weak"
    assert assistant.raw_data["runtime_policy_results"] == policies.results
    assert state.messages[-1] is assistant


def test_tau_agent_pre_weak_policy_falls_back_to_weak_call(tmp_path: Path, monkeypatch) -> None:
    policies = FakePolicies([{"requires_tool": False, "reason": "no deterministic tool"}])
    agent = _make_tau_agent(tmp_path, monkeypatch, policies)
    calls = []

    def fake_complete_sync(client, messages, temperature=0.2):
        calls.append(messages)
        return "I can help with that."

    monkeypatch.setattr("agentdistill.tau_bench._complete_sync", fake_complete_sync)
    state = agent.get_init_state()

    assistant, _ = agent.generate_next_message(FakeMessage(role="user", content="Hello."), state)

    assert assistant.content == "I can help with that."
    assert assistant.tool_calls is None
    assert len(calls) == 1
    assert len(policies.payloads) == 2
    assert policies.payloads[0]["runtime_policy_phase"] == "pre_weak"
    assert policies.payloads[1]["initial_answer"] == "I can help with that."


def test_tau_agent_post_weak_policy_denies_proposed_tool_call(tmp_path: Path, monkeypatch) -> None:
    policies = FakePolicies(
        [
            {
                "deny_tool": True,
                "tool_name": "cancel_reservation",
                "assistant_response": "I cannot cancel this reservation because it is not refundable under policy.",
                "reason": "cancellation eligibility not met",
            }
        ]
    )
    agent = _make_tau_agent(tmp_path, monkeypatch, policies)

    def fake_complete_sync(client, messages, temperature=0.2):
        return '{"tool_call": {"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}}'

    monkeypatch.setattr("agentdistill.tau_bench._complete_sync", fake_complete_sync)
    state = agent.get_init_state()

    assistant, _ = agent.generate_next_message(FakeMessage(role="user", content="Please cancel my reservation."), state)

    assert assistant.content == "I cannot cancel this reservation because it is not refundable under policy."
    assert assistant.tool_calls is None
    assert assistant.raw_data["runtime_policy_phase"] == "post_weak"
    assert assistant.raw_data["runtime_policy_denial"]["tool_name"] == "cancel_reservation"
    assert assistant.raw_data["runtime_policy_denial"]["tool_input"] == {"reservation_id": "MZDDS4"}
    assert len(policies.payloads) == 2
    assert policies.payloads[1]["tool_call"] == {"name": "cancel_reservation", "arguments": {"reservation_id": "MZDDS4"}}


def test_message_to_plain_dict_keeps_tool_calls() -> None:
    message = FakeMessage(
        role="assistant",
        tool_calls=[FakeToolCall(name="get_user", arguments={"user_id": "u1"})],
        turn_idx=2,
    )

    plain = message_to_plain_dict(message)

    assert plain["role"] == "assistant"
    assert plain["tool_calls"] == [{"id": "call_1", "name": "get_user", "arguments": {"user_id": "u1"}, "requestor": "assistant"}]
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
            "raw_data": None,
        }
    ]


def test_task_error_to_trace_records_adapter_error() -> None:
    task = type("Task", (), {"id": "task_1"})()

    trace = task_error_to_trace(task, domain="airline", split="train", exc=ValueError("bad response"))

    assert trace["task_id"] == "task_1"
    assert trace["termination_reason"] == "adapter_error"
    assert trace["reward_info"] is None
    assert trace["error"] == {"type": "ValueError", "message": "bad response"}


def test_load_tau_tasks_selects_requested_ids_in_requested_order() -> None:
    tau = {"registry": FakeTauRegistry([FakeTask("0"), FakeTask("1"), FakeTask("3")])}

    tasks = _load_tau_tasks(
        tau,
        domain="airline",
        split="train",
        task_set_name=None,
        num_tasks=2,
        task_ids=["3", "1"],
    )

    assert [task.id for task in tasks] == ["3", "1"]


def test_load_tau_tasks_rejects_unknown_task_ids() -> None:
    tau = {"registry": FakeTauRegistry([FakeTask("0")])}

    try:
        _load_tau_tasks(
            tau,
            domain="airline",
            split="train",
            task_set_name=None,
            num_tasks=1,
            task_ids=["missing"],
        )
    except RuntimeError as exc:
        assert "Unknown tau-bench task ids" in str(exc)
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


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


def test_build_tau_failure_digest_exposes_timeout_mutation_and_policy_counts() -> None:
    trace = {
        "domain": "airline",
        "split": "train",
        "task_id": "39",
        "termination_reason": "timeout",
        "reward_info": {"reward": 0.0},
        "messages": [
            {"role": "user", "content": "Cancel every upcoming flight for user amelia_davis_8890."},
            {
                "role": "assistant",
                "tool_calls": [{"name": "get_user_details", "arguments": {"user_id": "amelia_davis_8890"}}],
                "raw_data": {
                    "runtime_policy_results": [
                        {
                            "policy": "candidate_state",
                            "requires_tool": True,
                            "tool_name": "get_user_details",
                            "tool_input": {"user_id": "amelia_davis_8890"},
                        }
                    ]
                },
            },
            {"role": "tool", "content": '{"reservations": ["8C8K4E"]}'},
            {
                "role": "assistant",
                "tool_calls": [{"name": "cancel_reservation", "arguments": {"reservation_id": "8C8K4E"}}],
            },
        ],
    }

    digest = build_tau_failure_digest([trace])

    assert digest["failure_mode_counts"]["timeout"] == 1
    assert digest["failure_mode_counts"]["mutating_tool_call"] == 1
    assert digest["runtime_policy_counts"] == {"candidate_state": 1}
    failure = digest["traces"][0]
    assert failure["mutating_tool_call_names"] == ["cancel_reservation"]
    assert failure["runtime_policy_counts"] == {"candidate_state": 1}
    assert "progress controller" in digest["teacher_guidance"]["progress_controller_note"]


def test_build_tau_failure_digest_does_not_mark_successful_handoff_as_failure() -> None:
    trace = {
        "domain": "airline",
        "split": "train",
        "task_id": "49",
        "termination_reason": "user_stop",
        "reward_info": {"reward": 1.0},
        "messages": [
            {"role": "user", "content": "Please review this with a person."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "transfer_to_human_agents",
                        "arguments": {"summary": "User asks for human review."},
                    }
                ],
            },
        ],
    }

    digest = build_tau_failure_digest([trace])

    failure = digest["traces"][0]
    assert failure["mutating_tool_call_names"] == ["transfer_to_human_agents"]
    assert "mutating_tool_call" not in failure["failure_modes"]
    assert "mutating_tool_call" not in digest["failure_mode_counts"]


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
    assert "tau_runtime_evidence" in context


def test_build_tau_runtime_evidence_keeps_failed_tail_and_policy_phase() -> None:
    trace = {
        "domain": "airline",
        "split": "train",
        "task_id": "39",
        "termination_reason": "timeout",
        "reward_info": {"reward": 0.0},
        "messages": [
            {"role": "user", "content": "Cancel every upcoming flight.", "turn_idx": 1},
            {
                "role": "assistant",
                "content": "I will check your remaining reservations.",
                "turn_idx": 2,
                "raw_data": {
                    "runtime_policy_phase": "post_weak",
                    "runtime_policy_results": [{"policy": "candidate_state", "requires_tool": False}],
                    "runtime_policy_denial": {
                        "deny_tool": True,
                        "tool_name": "cancel_reservation",
                        "assistant_response": "I cannot cancel this.",
                    },
                },
            },
            {"role": "user", "content": "Please continue.", "turn_idx": 3},
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

    evidence = build_tau_runtime_evidence([trace], max_windows_per_trace=1)

    assert evidence["trace_windows"][0]["window_reason"] == "failed_trace_tail"
    tail_messages = evidence["trace_windows"][0]["messages"]
    raw = tail_messages[-1]["raw_data"]
    assert raw["runtime_policy_phase"] == "post_weak"
    assert raw["runtime_policy_results"] == [{"policy": "candidate_state", "requires_tool": False}]


def test_build_tau_runtime_evidence_includes_real_message_windows() -> None:
    trace = {
        "domain": "airline",
        "split": "train",
        "task_id": "1",
        "termination_reason": "max_steps",
        "reward_info": {"reward": 0.0},
        "messages": [
            {"role": "user", "content": "My user ID is raj_sanchez_7340.", "turn_idx": 3},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "get_user_details",
                        "arguments": {"user_id": "raj_sanchez_7340"},
                        "requestor": "assistant",
                    }
                ],
                "turn_idx": 4,
                "raw_data": {
                    "harnessforge_raw": '{"tool_call":{"name":"get_user_details"}}',
                    "runtime_policy_results": [
                        {
                            "policy": "tau_airline_candidate_state",
                            "requires_tool": True,
                            "tool_name": "get_user_details",
                            "tool_input": {"user_id": "raj_sanchez_7340"},
                        }
                    ],
                },
            },
            {
                "role": "tool",
                "content": '{"user_id": "raj_sanchez_7340", "reservations": ["MZDDS4", "60RX9E"]}',
                "requestor": "assistant",
                "error": False,
                "turn_idx": 5,
            },
        ],
    }

    evidence = build_tau_runtime_evidence([trace])

    assert evidence["schema"] == "harnessforge.tau_runtime_evidence.v1"
    assert evidence["message_shape_notes"]
    window = evidence["trace_windows"][0]
    assert window["window_reason"] == "runtime_policy_forced_tool"
    assert window["task_id"] == "1"
    assert window["messages"][1]["raw_data"]["runtime_policy_results"][0]["tool_input"] == {
        "user_id": "raj_sanchez_7340"
    }
    assert window["messages"][2]["role"] == "tool"
    assert "reservations" in window["messages"][2]["content"]
