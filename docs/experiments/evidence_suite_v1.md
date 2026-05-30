# Evidence Suite V1

Date: 2026-05-30

This records the first small evidence suite for HarnessForge. The goal is to summarize representative harness-transfer runs across different artifact types, instead of treating each benchmark as an isolated anecdote.

## Scope

The first suite uses two completed cloud runs:

```bash
python -m agentdistill.evidence --format markdown \
  outputs/benchmark_table_join_hard/default/table_join_hard_onepass_activation_hints_retry_v2 \
  outputs/benchmark_structured_extraction_focused/default/structured_extraction_v2
```

## Suite Summary

```text
runs = 2
end_to_end_transfer_runs = 2
end_to_end_runtime_transfer_runs = 1
end_to_end_harness_transfer_runs = 1
dev improved/regressed = 5/0
blind improved/regressed = 4/0
blind improved with runtime effect = 2
```

| run | status | artifacts | accepted | dev +/- | blind +/- | blind runtime effects | blind runtime wins |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| table_join_hard_onepass_activation_hints_retry_v2 | end_to_end_runtime_transfer | runtime_policy, test, tool | 2 | 3/0 | 2/0 | 2 | 2 |
| structured_extraction_v2 | end_to_end_harness_transfer | skill | 1 | 2/0 | 2/0 | 0 | 0 |

## Interpretation

`table_join_hard_onepass_activation_hints_retry_v2` is the runtime-control evidence case. The weak model failed both blind table-join aggregation tasks before harness evolution. The teacher generated a deterministic table calculator, a runtime policy, and regression tests. After harness evolution, both blind tasks triggered the policy and tool, and both became correct.

`structured_extraction_v2` is the non-runtime harness evidence case. The accepted teacher patch was a skill, not a tool or runtime policy. It still improved both dev and blind probes, so the evidence status is `end_to_end_harness_transfer` rather than `runtime_artifact_not_triggered`. This distinction matters because runtime policies are only one possible harness mechanism.

## Status Labels

The evidence reporter now separates two positive outcomes:

```text
end_to_end_runtime_transfer
```

Accepted harness changes produce blind improvement with an observed runtime effect such as a tool call or forced runtime policy.

```text
end_to_end_harness_transfer
```

Accepted harness changes produce blind improvement without a tool/runtime-policy activation. This covers skill, guideline, validator, or prompt-adjacent harness improvements.

## Why This Matters

This suite is a better project-level signal than a single benchmark run:

- it shows both executable runtime harness changes and skill-based harness changes
- it keeps dev and blind transfer separate
- it records whether blind gains coincide with runtime behavior changes
- it gives README and future papers a compact, reproducible evidence table

The suite is intentionally small. The next expansion should add one unstable or negative representative run, so the public report does not only contain successful examples.
