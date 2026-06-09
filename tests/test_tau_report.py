from __future__ import annotations

import json
from pathlib import Path

from agentdistill.report import build_tau_run_report


def test_tau_run_report_flags_official_pass_with_action_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "tau_run"
    run_dir.mkdir()
    summary = {
        "domain": "airline",
        "split": "train",
        "task_ids": ["41"],
        "traces": [
            {
                "task_id": "41",
                "termination_reason": "user_stop",
                "path": "41.json",
                "reward_info": {
                    "reward": 1.0,
                    "action_checks": [
                        {
                            "action": {
                                "action_id": "41_0",
                                "name": "get_user_details",
                                "arguments": {"user_id": "u1"},
                            },
                            "action_match": True,
                            "tool_type": "read",
                        },
                        {
                            "action": {
                                "action_id": "41_1",
                                "name": "get_reservation_details",
                                "arguments": {"reservation_id": "R1"},
                            },
                            "action_match": False,
                            "tool_type": "read",
                        },
                    ],
                },
            }
        ],
    }
    trace = {
        "task_id": "41",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "get_user_details",
                        "arguments": {"user_id": "u1"},
                    }
                ],
            }
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "41.json").write_text(json.dumps(trace), encoding="utf-8")

    report = build_tau_run_report(run_dir, output_path=run_dir / "tau_report.json")

    assert (run_dir / "tau_report.json").exists()
    assert report["aggregate"]["official_passed"] == 1
    assert report["aggregate"]["official_pass_but_action_mismatch"] == 1
    assert report["aggregate"]["strict_action_unmatched_tasks"] == 1
    assert report["tasks"][0]["strict_action_pass"] is False
    assert report["tasks"][0]["official_pass_but_action_mismatch"] is True
    assert report["tasks"][0]["unmatched_actions"] == [
        {
            "action_id": "41_1",
            "name": "get_reservation_details",
            "arguments": {"reservation_id": "R1"},
            "tool_type": "read",
        }
    ]


def test_tau_run_report_counts_write_tool_calls(tmp_path: Path) -> None:
    run_dir = tmp_path / "tau_run"
    run_dir.mkdir()
    summary = {
        "domain": "airline",
        "split": "train",
        "task_ids": ["39"],
        "traces": [
            {
                "task_id": "39",
                "termination_reason": "user_stop",
                "path": "39.json",
                "reward_info": {
                    "reward": 1.0,
                    "action_checks": [
                        {
                            "action": {
                                "action_id": "39_0",
                                "name": "cancel_reservation",
                                "arguments": {"reservation_id": "R1"},
                            },
                            "action_match": True,
                            "tool_type": "write",
                        }
                    ],
                },
            }
        ],
    }
    trace = {
        "task_id": "39",
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "cancel_reservation",
                        "arguments": {"reservation_id": "R1"},
                    }
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "transfer_to_human_agents",
                        "arguments": {"summary": "done"},
                    }
                ],
            },
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "39.json").write_text(json.dumps(trace), encoding="utf-8")

    report = build_tau_run_report(run_dir)

    assert report["aggregate"]["expected_write_actions"] == 1
    assert report["aggregate"]["matched_write_actions"] == 1
    assert report["aggregate"]["actual_write_tool_calls"] == 2
    assert report["tasks"][0]["actual_write_tool_names"] == ["cancel_reservation", "transfer_to_human_agents"]
    assert report["tasks"][0]["actual_cancel_reservation_ids"] == ["R1"]
