# Rejection-Aware Repair Success

Date: 2026-05-15

This records the first run where a teacher-generated code harness bundle was rejected by contracts, repaired by the teacher in the next iteration, accepted, and then improved both teacher-visible dev probes and blind final tests.

## Run

Server workspace:

```text
/auxstore/cse/d11/data/s1155124388/projects/AgentDistill_repair_success_clean
```

Command:

```bash
REQUEST_TIMEOUT_SECONDS=120 TEACHER_TIMEOUT_SECONDS=300 WEAK_TIMEOUT_SECONDS=120 \
  ../AgentDistill_bench_clean/.venv/bin/python -m agentdistill.benchmark \
  --config configs/benchmark_inventory.yaml \
  --run-id repair_success_v1
```

The workspace was a clean clone before the run. Teacher-generated harness files were left untracked after the run and were not committed as baseline code.

## Result

Patch summary:

```text
train_steps = 3
accepted = 2
rejected = 1
accepted_code_manifest_bundles = 1
contract_failures = 2
```

Transfer:

```text
dev_transfer.improved = 2
dev_transfer.regressed = 0
blind_transfer.improved = 2
blind_transfer.regressed = 0
```

The blind final test improved on both tasks:

```text
blind_inventory_badges:   false -> true
blind_inventory_vouchers: false -> true
```

## Repair Sequence

Iteration 1 proposed code bundle:

```text
bundle_id = inventory_arithmetic_policy_v1
status = rejected
```

The tool tests passed, but the runtime policy failed. It classified shipped/sold quantities as additions instead of subtractions:

```text
expected additions:    [666, 322]
expected subtractions: [89, 647]
actual additions:      [666, 322, 647]
actual subtractions:   [89]
```

The contract gate rejected the whole bundle before it reached the real harness.

Iteration 2 repaired the same bundle id:

```text
bundle_id = inventory_arithmetic_policy_v1
status = accepted
```

Accepted artifacts:

```text
harness/tools/inventory_arithmetic.py
harness/tests/inventory_arithmetic.json
harness/runtime_policies/force_inventory_arithmetic.py
harness/tests/force_inventory_arithmetic.json
```

The accepted contracts were:

```text
manifest matches patch bundle
tool test file exists and all cases passed
forced tool call succeeded
all policy tests passed
```

Iteration 3 added a prompt-only final-format guideline:

```text
harness/guidelines/inventory_final_format.md
```

## Why The Repair Worked

The teacher received structured `patch_feedback` from the rejected bundle in the next training iteration. The feedback included:

```text
bundle_id
rejection_reason
rejected_patch_paths
failed_contracts
expected vs actual tool_input
tool_result
```

The second iteration result file records `context_patch_feedback`, proving that the teacher received the rejected contract summary before proposing the repair.

The teacher also had a canonical self-contained inventory parser scaffold in the diagnosis prompt. That scaffold covered:

```text
comma-formatted numbers
product terms
operation classification
```

## Why This Matters

This run demonstrates the full intended loop:

```text
teacher proposes code harness
-> contract gate rejects a bad parser
-> rejection summary enters benchmark_context
-> teacher repairs the same bundle
-> contracts accept the repaired harness
-> weak-only dev and blind tests improve
```

The weak model weights were unchanged. The improvement came from a teacher-generated tool, runtime policy, and tests that changed the weak model's operating environment.
