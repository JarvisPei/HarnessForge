# Repair Efficiency Ablation v1

Date: 2026-05-16

This records a small controlled ablation on `benchmark_explicit_ops_v2` using the repair-efficiency telemetry added in commit `5a5cc06` and benchmark override CLI added in commit `dadef6e`.

## Setup

Server workspace:

```text
/auxstore/cse/d11/data/s1155124388/projects/AgentDistill_scoped_repair_verify
```

All ablation runs used:

```bash
REQUEST_TIMEOUT_SECONDS=240 TEACHER_TIMEOUT_SECONDS=600 WEAK_TIMEOUT_SECONDS=300 \
  ../AgentDistill_bench_clean/.venv/bin/python -m agentdistill.benchmark \
  --config configs/benchmark_explicit_ops_v2.yaml \
  --evolve-iterations 2
```

Variants:

```text
ablate_full_train_v1:    --repair-mode full_train --inner-repair-attempts 0
ablate_focused_only_v1:  --repair-mode focused    --inner-repair-attempts 0
ablate_scoped_inner_v1:  --repair-mode focused    --inner-repair-attempts 1
```

The scoped-inner run was interrupted during `evolve_train_iter_02/train_explicit_cards_jsonish` after a long API wait. The repair-efficiency report recovered available telemetry from phase outputs, so patch/repair efficiency is usable, but final dev/blind impact is not comparable for that run.

Aggregate command:

```bash
../AgentDistill_bench_clean/.venv/bin/python -m agentdistill.repair_efficiency \
  outputs/benchmark_explicit_ops_v2/default/ablate_full_train_v1 \
  outputs/benchmark_explicit_ops_v2/default/ablate_focused_only_v1 \
  outputs/benchmark_explicit_ops_v2/default/ablate_scoped_inner_v1 \
  -o outputs/benchmark_explicit_ops_v2/default/ablation_repair_efficiency_v1.json
```

## Results

```text
variant              accepted_rate  accepted/rejected  inner  scoped_inner  paths  unique_paths  weak_skipped
full_train           0.3333         1 / 2              0      0             8      6             0
focused_only         0.3333         1 / 2              0      0             10     5             0
scoped_inner         0.6667         2 / 1              1      1             7      7             1
```

Transfer movement for completed runs:

```text
full_train:   dev improved=0 regressed=0, blind improved=0 regressed=0
focused_only: dev improved=0 regressed=0, blind improved=0 regressed=0
```

The scoped-inner run does not have final dev/blind rows because it was interrupted before the final probes. Its patch telemetry still shows the relevant mechanism signal:

```text
patch_attempts = 3
accepted = 2
rejected = 1
accepted_rate = 0.6667
inner_repair_attempts = 1
scoped_inner_repair_attempts = 1
total_patch_paths = 7
focused_repair_weak_calls_skipped = 1
```

Historical reference, not a strict ablation:

```text
/auxstore/cse/d11/data/s1155124388/projects/AgentDistill_inner_repair_clean/outputs/benchmark_explicit_ops_v2/default/explicit_ops_v2_inner_repair_v1
accepted_rate = 0.6
dev improved = 1, regressed = 0
blind improved = 2, regressed = 0
```

That run predates the current scoped telemetry, so inner/scoped counts are not directly comparable.

## Interpretation

The useful signal is repair efficiency, not final benchmark accuracy:

```text
scoped_inner accepted_rate > full_train and focused_only
scoped_inner touched fewer total patch paths
scoped_inner recorded a real scoped inner repair attempt
```

This supports the hypothesis that artifact-scoped inner repair makes the harness evolution loop more contained and more likely to produce accepted patches. It does not yet prove stronger transfer, because the scoped-inner run was interrupted before final probes and the two completed ablations showed no dev/blind movement.

## Caveats

This was one run per condition and API stochasticity is high. The scoped-inner condition is partial. Full statistical confidence would require repeated runs or shorter deterministic benchmark slices.

The `focused_only` run did not get a focused repair phase benefit in this two-iteration setup unless a previous rejected bundle made the next iteration focused. In practice, focused repair is most informative in longer runs where rejection feedback carries into later iterations.

## Next Step

The next experiment should use a shorter benchmark slice designed to finish reliably and force one rejection, so all variants can complete final dev/blind probes. The current ablation is still useful because it validates that the telemetry exposes the right mechanism-level signal.
