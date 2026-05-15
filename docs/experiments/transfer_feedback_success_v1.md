# Transfer Feedback Harness Repair Success

Date: 2026-05-15

This records the first successful run where an accepted harness failed on dev transfer, the failure was summarized as teacher feedback, and a later teacher iteration repaired the executable harness without a hand-coded benchmark-specific parser fix.

## Run

Server workspace:

```text
/auxstore/cse/d11/data/s1155124388/projects/AgentDistill_transfer_feedback_only_clean
```

Command:

```bash
REQUEST_TIMEOUT_SECONDS=180 TEACHER_TIMEOUT_SECONDS=300 WEAK_TIMEOUT_SECONDS=300 \
  ../AgentDistill_bench_clean/.venv/bin/python -m agentdistill.benchmark \
  --config configs/benchmark_explicit_ops_v2.yaml \
  --run-id explicit_ops_v2_feedback_only_v1
```

The workspace was a clean clone at commit `97eb481`.

## Result

Patch summary:

```text
train_steps = 6
accepted = 4
rejected = 2
accepted_code_manifest_bundles = 4
accepted_tool_test_policy_bundles = 4
contract_failures = 4
```

Transfer:

```text
dev_transfer.before_success = 1
dev_transfer.after_success = 3
dev_transfer.improved = 2
dev_transfer.regressed = 0

blind_transfer.before_success = 0
blind_transfer.after_success = 2
blind_transfer.improved = 2
blind_transfer.regressed = 0
```

Final blind improvements:

```text
blind_explicit_badges_reordered_jsonish: false -> true
blind_explicit_tickets_semicolon:       false -> true
```

## Feedback Sequence

This run used:

```yaml
transfer_context_mode: feedback_only
```

That means the teacher did not see the full dev probe schema in the first train iteration. Dev probes were used only after accepted harnesses had been installed.

Iteration 1 accepted two teacher-generated harness bundles, but the accepted runtime policy was too narrow. On the first transfer probe, all three dev probe tasks still failed. The semicolon task was especially important because it regressed:

```text
dev_explicit_vouchers_semicolon
before_success = true
after_success = false
regressed = true
after_tool_call = null
after_tool_result = null
runtime_policy.requires_tool = false
```

The failed answer was:

```text
17,820 vouchers remain.
```

The expected answer was:

```text
16,870 vouchers remain.
```

Iteration 2 teacher diagnoses received `context_transfer_feedback` with the failed task instructions, expected answers, weak answers, tool calls, tool results, and runtime policy results. The teacher responded by proposing a broader executable repair:

```text
Revise explicit_inventory_calculator to parse JSON objects, line blocks,
inline key/value lists with start/initial aliases, semicolon-delimited
operations, and Markdown sign/value tables. Revise
force_explicit_inventory_calculator so it routes any task containing an
explicit unit, initial/start count, and signed operation list/table to the
tool with raw_text.
```

Those iteration 2 bundles were rejected by the contract gate.

Iteration 3 used the rejection feedback to repair the generated tool and runtime policy. The accepted harness then forced the calculator on the semicolon dev task:

```text
transfer_probe_iter_03/dev_explicit_vouchers_semicolon
tool = explicit_inventory_calculator
tool_result.ok = true
tool_result.result = 16870
runtime_policy.requires_tool = true
```

The final blind semicolon task also succeeded:

```text
after_blind_test/blind_explicit_tickets_semicolon
tool = explicit_inventory_calculator
tool_result.ok = true
tool_result.result = 26195
weak_answer = 26,195 tickets remain.
```

## Why This Matters

This run demonstrates the desired on-policy harness repair loop:

```text
teacher-generated harness
-> accepted by contracts
-> dev transfer probe exposes a real harness failure
-> transfer_feedback reaches the next teacher iteration
-> teacher proposes executable tool/runtime-policy repairs
-> contract gate rejects invalid repair attempts
-> patch_feedback reaches the next teacher iteration
-> teacher repairs the harness
-> weak model improves on clean blind tasks
```

The semicolon parser/router repair was not committed by hand. The committed framework changes only added the transfer feedback mechanism and a `feedback_only` benchmark mode so the teacher cannot solve dev transfer by seeing the dev schema before the first accepted harness is tested.

## Strategy Check

This matches the project strategy:

```text
frontier teacher architects harness
weak model runs with the improved harness
dev probe provides repair feedback
blind probe remains final evaluation
model weights stay unchanged
improvement comes from executable environment changes
```

## Next Signal

The next useful experiment is to make transfer feedback less all-or-nothing. In this run, iteration 2 correctly identified the needed repair but produced contract-invalid artifacts, and iteration 3 fixed them through patch feedback. A stronger loop would preserve the same transfer failure context across the patch-feedback repair step, so the teacher repairs the rejected bundle while still seeing the original dev transfer failures.
