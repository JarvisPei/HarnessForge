# Evidence Suite V2 Boundary Case

Date: 2026-05-30

This records the first evidence-suite expansion that intentionally includes a boundary case. The goal is to avoid a success-only suite and identify which part of the harness-distillation loop still fails under the current mainline.

## Command

The boundary run was executed on the cloud VM with a clean generated harness:

```bash
git clean -fd harness
git clean -fdX harness

REQUEST_TIMEOUT_SECONDS=1200 TEACHER_TIMEOUT_SECONDS=1200 WEAK_TIMEOUT_SECONDS=1200 \
REQUEST_MAX_RETRIES=2 REQUEST_RETRY_BACKOFF_SECONDS=5 \
TEACHER_MAX_RETRIES=5 TEACHER_RETRY_BACKOFF_SECONDS=10 \
  python -m agentdistill.benchmark \
  --config configs/benchmark_inventory_focused.yaml \
  --run-id inventory_focused_boundary_v1
```

The suite summary was generated with:

```bash
python -m agentdistill.evidence --format markdown \
  outputs/benchmark_table_join_hard/default/table_join_hard_onepass_activation_hints_retry_v2 \
  outputs/benchmark_structured_extraction_focused/default/structured_extraction_v2 \
  outputs/benchmark_inventory_focused/default/inventory_focused_boundary_v1
```

## Suite Summary

```text
runs = 3
end_to_end_transfer_runs = 2
end_to_end_runtime_transfer_runs = 1
end_to_end_harness_transfer_runs = 1
dev improved/regressed = 7/0
blind improved/regressed = 4/0
blind improved with runtime effect = 2
```

| run | status | artifacts | accepted | dev +/- | blind +/- | blind runtime effects | blind runtime wins |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| table_join_hard_onepass_activation_hints_retry_v2 | end_to_end_runtime_transfer | runtime_policy, test, tool | 2 | 3/0 | 2/0 | 2 | 2 |
| structured_extraction_v2 | end_to_end_harness_transfer | skill | 1 | 2/0 | 2/0 | 0 | 0 |
| inventory_focused_boundary_v1 | runtime_effect_without_blind_transfer | guideline, runtime_policy, test, tool | 3 | 2/0 | 0/0 | 2 | 0 |

## Boundary Run Result

`inventory_focused_boundary_v1` improved both teacher-visible dev probes:

```text
dev_improved = 2 / 2
dev_runtime_effect = 2 / 2
```

It did not improve blind probes:

```text
blind_improved = 0 / 2
blind_runtime_effect = 2 / 2
blind_improved_with_runtime_effect = 0 / 2
```

The teacher accepted three harness updates:

```text
harness/tools/inventory_arithmetic.py
harness/tests/inventory_arithmetic.json
harness/runtime_policies/force_inventory_arithmetic.py
harness/tests/force_inventory_arithmetic.json
harness/guidelines/finalize_count_result.md
```

## Diagnosis

This is not a routing failure. The runtime policy fired on both blind tasks, and the weak model called the generated tool.

The failure is tool semantic generalization. The generated tool learned enough subtraction verbs for dev transfer:

```text
threw away
sold
used
lost
```

but missed blind-only subtraction verbs:

```text
handed out
voided
redeemed
```

Observed blind failures:

```text
blind_inventory_badges
expected = 3323
tool_result = 4529
missing operation = subtract handed out 1206

blind_inventory_vouchers
expected = 1809
tool_result = 2909
missing operations = subtract voided 96, subtract redeemed 1004
```

## Interpretation

This is the right kind of boundary case for the current project stage:

- the teacher can generate executable harness artifacts
- contract checks accepted the bundle
- dev transfer improved
- blind runtime behavior changed
- blind accuracy did not improve because the tool's operation semantics were too narrow

The next mechanism improvement should not add inventory-specific verbs by hand. The better target is a teacher-facing generalization contract for semantic operation coverage: when a tool maps natural-language events to signed operations, tests should include teacher-generated paraphrase/adversarial verbs for both add and subtract classes before the bundle is accepted.
