# Repair Mechanism Benchmark v1

Date: 2026-05-16

This records the first attempt to create a short mechanism benchmark that can support repair-mode ablations at lower API cost than `benchmark_explicit_ops_v2`.

## Config

```text
configs/benchmark_repair_mechanism.yaml
```

The slice has:

```text
train_tasks = 1
dev_probe_tasks = 2
blind_test_tasks = 1
evolve_iterations = 2
transfer_context_mode = feedback_only
repair_mode = focused
inner_repair_attempts = 1
policy_generalization_audit = true
```

The task family is still explicit signed-operation counting, but shorter than `benchmark_explicit_ops_v2`. The intention was to make the run cheap while still encouraging a tool/runtime-policy harness and exposing parser/router gaps.

## Server Runs

Workspace:

```text
/auxstore/cse/d11/data/s1155124388/projects/AgentDistill_scoped_repair_verify
```

Smoke command:

```bash
REQUEST_TIMEOUT_SECONDS=240 TEACHER_TIMEOUT_SECONDS=600 WEAK_TIMEOUT_SECONDS=300 \
  ../AgentDistill_bench_clean/.venv/bin/python -m agentdistill.benchmark \
  --config configs/benchmark_repair_mechanism.yaml \
  --run-id mechanism_smoke_v2 \
  --evolve-iterations 2 \
  --repair-mode focused \
  --inner-repair-attempts 1
```

Result:

```text
patch_attempts = 2
accepted = 2
rejected = 0
accepted_rate = 1.0
inner_repair_attempts = 2
scoped_inner_repair_attempts = 2
dev improved = 2
blind improved = 0
```

Audit-enabled smoke command:

```bash
REQUEST_TIMEOUT_SECONDS=240 TEACHER_TIMEOUT_SECONDS=600 WEAK_TIMEOUT_SECONDS=300 \
  ../AgentDistill_bench_clean/.venv/bin/python -m agentdistill.benchmark \
  --config configs/benchmark_repair_mechanism.yaml \
  --run-id mechanism_audit_smoke_v1 \
  --evolve-iterations 1 \
  --repair-mode focused \
  --inner-repair-attempts 1
```

Result:

```text
patch_attempts = 1
accepted = 1
rejected = 0
accepted_rate = 1.0
inner_repair_attempts = 0
scoped_inner_repair_attempts = 0
dev improved = 0
blind improved = 0
```

## Interpretation

This is a useful short smoke benchmark, but it is not yet a reliable rejection-driven mechanism benchmark.

The teacher often produced policy/test-only harness updates that passed contracts. Even with `policy_generalization_audit = true`, the first patch was accepted in the audit-enabled run. That means the slice does not reliably exercise:

```text
rejected patch -> focused repair -> inner repair -> artifact-scoped repair
```

The positive signal is that the benchmark is cheap and can show transfer improvement. The negative signal is more important for the current goal: natural teacher behavior is too adaptive to guarantee a rejected patch with only task/config changes.

## Next Design Direction

To reliably expose repair-mode differences, the mechanism benchmark likely needs a controlled teacher fixture or replayed diagnosis, not only a natural frontier-teacher run. A fixture can produce the same intentionally invalid first patch across ablation modes, then the framework can measure whether focused repair, inner repair, and artifact scope recover faster with less file churn.

This would separate two benchmarks:

```text
natural mechanism smoke: cheap end-to-end teacher/harness sanity check
controlled repair fixture: deterministic ablation of repair-loop mechanics
```

The current config should remain as the natural smoke slice. The next step is to add a controlled repair fixture runner or test harness for deterministic rejected-patch ablations.
