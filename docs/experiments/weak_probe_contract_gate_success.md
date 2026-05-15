# Weak-Probe Contract-Gated Transfer Success

Date: 2026-05-15

This records the first positive held-out transfer result from the harness distillation loop.

## Run

Server workspace:

```text
/auxstore/cse/d11/data/s1155124388/projects/AgentDistill_transfer_clean
```

Command:

```bash
REQUEST_TIMEOUT_SECONDS=120 TEACHER_TIMEOUT_SECONDS=300 WEAK_TIMEOUT_SECONDS=120 \
  ../AgentDistill_bench_clean/.venv/bin/python -m agentdistill.benchmark \
  --config configs/benchmark_inventory.yaml \
  --run-id weak_probe_contract_gate_v1
```

The workspace was a clean clone before the run. Teacher-generated harness files were left untracked after the run and were not committed as baseline code.

## Result

Held-out impact:

```text
improved = 2
regressed = 0
```

Both held-out inventory tasks improved from incorrect baseline answers to correct final answers:

```text
heldout_inventory_tags:     false -> true
heldout_inventory_stickers: false -> true
```

## Harness Changes

The teacher produced a reusable bundle instead of only patching the training answer:

```text
harness/tools/inventory_arithmetic.py
harness/tests/inventory_arithmetic.json
harness/runtime_policies/force_inventory_arithmetic.py
harness/guidelines/inventory_arithmetic.md
```

The accepted bundle included:

```text
tool test cases: 3
runtime policy contract: forced tool result matched the training expected answer
```

## Probe Semantics

Held-out probe phases evaluated the weak model only:

```text
baseline_heldout:       patch_path = null
transfer_probe_iter_01: patch_path = null
transfer_probe_iter_02: patch_path = null
after_heldout:          patch_path = null
```

The teacher still received feedback through the next training diagnosis via `benchmark_context`, but held-out examples were not patched directly.

## Why This Matters

The result demonstrates the intended mechanism:

```text
teacher-generated harness bundle -> weak model behavior changes -> held-out transfer improves
```

The model weights were unchanged. The improvement came from a teacher-generated tool, tests, runtime policy, and guideline that made the weak model use deterministic arithmetic on related held-out tasks.

## Guardrail Added

A previous run accepted bad runtime policies because the policy only had to return a tool result with `ok: true`. The new contract rejects forced tool calls whose result does not contain the expected training answer number.

This prevents the teacher from installing a policy that is syntactically valid but semantically wrong on the training contract.
