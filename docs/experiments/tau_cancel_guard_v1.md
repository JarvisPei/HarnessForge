# Tau Cancellation Guard v1

Date: 2026-06-08

## Why This Iteration Mattered

The previous tau airline task 1 run reached the correct reservation through the
teacher-generated candidate-state policy, then failed because the weak model
miscomputed the 24-hour cancellation window and called `cancel_reservation`.
The architect decision was that the existing runtime policy boundary was
insufficient: replacing a destructive tool call with another tool could avoid
one mutation, but it could not give the user a policy-faithful denial.

This led to a framework change rather than a hand-written benchmark fix:
post-weak tau runtime policies can now return `deny_tool=true` with an
`assistant_response`. The adapter suppresses the proposed weak tool call and
emits the response as a normal assistant message.

Framework commit:

```text
618ab3a Add tau runtime policy tool denial
```

## Teacher-Generated Harness

After the framework change, the teacher generated:

```text
harness/runtime_policies/tau_airline_cancel_guard.py
harness/tests/tau_airline_cancel_guard.json
```

These files were applied only on the cloud active harness workspace and remain
untracked experiment artifacts. The teacher output needed mechanical manifest
completion and test-schema normalization (`expect` to `expected`), but the
runtime policy and test semantics were not hand-written.

Atomic gate result:

```text
patch_status: accepted
runtime policy tests: 4 passed
```

Important artifacts:

```text
outputs/tau_bench_teacher_probe/airline_task1_cancel_guard_patch_v2_short/
  patch_raw.json
  patch_parsed.json
  apply_result_manifest_tests_completed.json
  replay_old_failure_policy_results.json
  replay_old_failure_adapter_decision.json
```

## Real Tau Result

Command:

```bash
TAU2_DATA_DIR=$HOME/projects/tau2-bench/data \
.venv/bin/python -m agentdistill.tau_bench \
  --domain airline --split train --task-id 1 --num-tasks 1 \
  --user-llm gpt-5.5 --user-llm-shim \
  --output-dir outputs/tau_bench_policy/airline_train_1_cancel_guard_v1 \
  --max-steps 24 --max-errors 3 --timeout 600
```

Result:

```text
termination_reason: user_stop
reward: 1.0
duration: 37.85s
```

Interpretation detail: in this successful real run, the cancellation guard did
not need to fire. The weak model, after the route-search policy found Q69X3R,
declined the cancellation/refund itself. This means the real-run reward should
not be overclaimed as direct evidence that the guard fired in the live trace.

## Replay Evidence

The stronger causal check used the previous failed trace, where the weak model
did propose:

```json
{"tool_call": {"name": "cancel_reservation", "arguments": {"reservation_id": "Q69X3R"}}}
```

With the new teacher-generated guard active, replaying that exact pre-tool
state produced:

```json
{
  "deny_tool": true,
  "tool_name": "cancel_reservation",
  "tool_input": {"reservation_id": "Q69X3R"}
}
```

The adapter selector returned:

```text
tool_payloads: []
denial: present
```

So the new framework permission plus teacher-generated harness would suppress
the destructive tool call on the observed failure trajectory.

## Small Cancellation Slice

After the guard was active, a small train cancellation slice was run:

```bash
TAU2_DATA_DIR=$HOME/projects/tau2-bench/data \
.venv/bin/python -m agentdistill.tau_bench \
  --domain airline --split train \
  --task-id 0 --task-id 1 --task-id 28 --num-tasks 3 \
  --user-llm gpt-5.5 --user-llm-shim \
  --output-dir outputs/tau_bench_policy/airline_cancel_slice_0_1_28_v1 \
  --max-steps 30 --max-errors 3 --timeout 600
```

Result:

```text
task 0: reward=1.0, termination=user_stop, cancel_calls=0, guard_denials=0
task 1: reward=1.0, termination=user_stop, cancel_calls=0, guard_denials=0
task 28: reward=1.0, termination=user_stop, cancel_calls=0, guard_denials=0
```

Interpretation detail: this is a useful non-regression and transfer-adjacent
signal, but not direct live evidence for the denial guard. In all three tasks,
the weak model avoided `cancel_reservation` by itself after the harness helped
it retrieve the relevant state. The guard was evaluated multiple times, but it
did not need to deny a proposed cancellation.

Slice artifact:

```text
outputs/tau_bench_policy/airline_cancel_slice_0_1_28_v1/harnessforge_slice_analysis.json
```

## Replay Ablation

A controlled replay ablation compared the previous failed pre-tool state under
four policy sets:

```text
none:                 would_execute_cancel=true,  would_suppress_tool=false
candidate_only:       would_execute_cancel=true,  would_suppress_tool=false
guard_only:           would_execute_cancel=false, would_suppress_tool=true
candidate_plus_guard: would_execute_cancel=false, would_suppress_tool=true
```

This isolates the guard's marginal effect. The candidate-state policy gets the
agent to the right reservation, but once the weak model proposes
`cancel_reservation`, candidate-state alone does not block it. The
teacher-generated cancel guard is the artifact that changes the post-weak
tool boundary.

Ablation artifact:

```text
outputs/tau_bench_teacher_probe/airline_task1_cancel_guard_ablation_v1/ablation_summary.json
```

## Takeaway

This is a useful method signal: the teacher did not merely tune a prompt. It
identified a missing control surface, the framework exposed that control surface,
and the teacher then generated a tested harness artifact that changes the
adapter's action boundary.

The live train slice suggests the active harness is not regressing nearby
cancellation tasks. The strongest causal evidence for the new guard is the
replay ablation: route/candidate-state alone still executes the destructive
tool, while guard-only and guard-plus-candidate suppress it. The next step
should search for cancellation tasks or seeds where weak proposes
`cancel_reservation` in live runs, so the denial path is exercised inside a full
tau trajectory rather than only replay.
