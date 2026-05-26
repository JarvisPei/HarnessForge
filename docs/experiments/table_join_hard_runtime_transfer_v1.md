# Table Join Hard Runtime Transfer

Date: 2026-05-26

This records the first strong end-to-end evidence case where the teacher acted as a harness architect and improved weak-model blind performance through generated executable runtime artifacts.

## Command

Run on the cloud VM from the canonical checkout:

```bash
REQUEST_TIMEOUT_SECONDS=1200 TEACHER_TIMEOUT_SECONDS=1200 WEAK_TIMEOUT_SECONDS=1200 \
  python -m agentdistill.benchmark \
  --config configs/benchmark_table_join_hard.yaml \
  --run-id table_join_hard_runtime_effect_v1
```

Before the run, generated harness artifacts were cleared with:

```bash
git clean -fd harness
```

## Result

The run completed end to end:

```text
baseline_dev_probe
baseline_blind_test
evolve_train_iter_01
transfer_probe_iter_01
evolve_train_iter_02
transfer_probe_iter_02
evolve_train_iter_03
transfer_probe_iter_03
after_dev_probe
after_blind_test
```

Key metrics:

```text
accepted = 2
rejected = 1
accepted_runtime_artifact = 2
accepted_test_only = 0
accepted_but_no_runtime_artifact = 0

dev_improved = 2 / 2
blind_improved = 2 / 2
dev_runtime_effect = 2 / 2
blind_runtime_effect = 2 / 2
blind_improved_with_runtime_effect = 2 / 2
```

## Generated Harness

The teacher generated a bundle with executable runtime artifacts:

```text
harness/tools/table_lookup_margin.py
harness/runtime_policies/force_table_lookup_margin.py
harness/tests/table_lookup_margin.json
harness/tests/force_table_lookup_margin.json
```

The accepted bundle included a tool, runtime policy, and tests. It was not a prompt-only or test-only patch.

## Blind Evidence

Before harness evolution, the weak model failed both blind tasks:

```text
blind_north_completed_margin: expected 395, weak answered 463
blind_priority_done_margin: expected 406, weak answered 66
```

After harness evolution, both blind tasks triggered the generated runtime policy and tool:

```text
after_runtime_policy_fired = true
after_runtime_effect = tool_call
tool_name = table_lookup_margin
```

The tool results matched the expected answers:

```text
blind_north_completed_margin -> 395
blind_priority_done_margin -> 406
```

## Interpretation

This is a clean positive result for the core project hypothesis:

```text
weak baseline failure
-> teacher diagnoses missing deterministic table-join aggregation support
-> teacher generates tool + runtime policy + tests
-> runtime policy routes blind tasks to the generated tool
-> weak model answers correctly using the tool result
```

The important point is not only that blind accuracy improved. The impact report shows that the improvement coincided with actual runtime behavior changes: both blind improvements had runtime policy activation and tool calls.

## Follow-Up

This family should become the first evidence benchmark. Future runs should be summarized with:

```bash
python -m agentdistill.evidence <run_dir> [...]
```

The key stability questions are:

```text
Does the teacher consistently generate runtime artifacts?
Does the generated runtime policy consistently trigger on schema-renamed blind tasks?
Does blind_improved_with_runtime_effect remain positive across clean repeats?
```

## Repeatability Check

Two clean repeats were run after adding the evidence reporter:

```bash
python -m agentdistill.evidence \
  outputs/benchmark_table_join_hard/default/table_join_hard_runtime_effect_v1 \
  outputs/benchmark_table_join_hard/default/table_join_hard_runtime_effect_v2 \
  outputs/benchmark_table_join_hard/default/table_join_hard_runtime_effect_v3
```

Aggregate evidence:

```text
num_runs = 3
runtime_artifact_runs = 3
runtime_effect_runs = 3
blind_runtime_effect_runs = 3
end_to_end_transfer_runs = 1
blind_improved_with_runtime_effect = 2
```

Status counts:

```text
end_to_end_transfer = 1
runtime_effect_without_blind_transfer = 2
```

This changes the interpretation from "solved" to "mechanism established, transfer unstable." The teacher reliably produced or retained runtime-affecting harness artifacts, and blind tasks did trigger runtime behavior in all three runs. The main instability is inside the generated tool's schema generalization:

```text
v2 blind failure: tool returned "required tables or columns not found"
v3 blind failure: tool returned "could not identify detail table"
```

The next project step should target generated-tool generalization, especially table parser/schema inference, rather than more evidence instrumentation.
