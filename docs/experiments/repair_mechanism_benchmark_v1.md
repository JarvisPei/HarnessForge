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

This fixture benchmark reliably triggers a rejected harness patch in all three modes:

```text
fixture_full_train:   rejected = 1
fixture_focused_only: rejected = 1
fixture_scoped_inner: rejected = 1
```

The important difference is that the scoped-inner variant still records an inner repair attempt:

```text
fixture_scoped_inner: inner_repair_attempts = 1
```

The benchmark is cheap and deterministic, but the current controlled fixture does not guarantee inner repair success. It still exposes the repair-loop mechanics without API cost:

```text
natural teacher benchmark: no reliable rejection
controlled fixture: always rejected initial patch, scoped-inner path exercised
```

## Next Design Direction

To compare focused repair, inner repair, and artifact-scoped repair more sharply, the next iteration should improve the fixture so the scoped-inner attempt succeeds while baseline/focused-only still reject. That would convert this from a rejection smoke into a full ablation probe.

This would separate two benchmarks:

```text
natural mechanism smoke: cheap end-to-end teacher/harness sanity check
controlled repair fixture: deterministic ablation of repair-loop mechanics
```

The current config remains a natural smoke slice, but the deterministic fixture runner is the real benchmark object for rejection-driven repair ablations.

## Mechanism Fixture v2

The deterministic fixture now reports repair outcome separately from outer patch acceptance. This matters because a scoped-inner run should still show the initial bad patch as rejected, while also showing that the inner scoped repair accepted a corrected harness patch.

Validation command:

```bash
.venv/bin/python -m agentdistill.repair_fixture --output-dir /private/tmp/repair_fixture_v2
```

Result:

```text
fixture_full_train:
  rejected = 1
  repair_success = false
  repair_success_via = none
  scoped_inner_repair_success = false

fixture_focused_only:
  rejected = 1
  repair_success = false
  repair_success_via = none
  scoped_inner_repair_success = false

fixture_scoped_inner:
  rejected = 1
  repair_success = true
  repair_success_via = scoped_inner_repair
  scoped_inner_repair_success = true
```

Aggregate:

```text
rejected = 3
repair_successes = 1
inner_repair_accepted = 1
scoped_inner_repair_accepted = 1
scoped_inner_repair_successes = 1
```

This converts the fixture into a small mechanism-differentiating ablation: all variants reject the same initial bad runtime-policy patch, but only the scoped-inner variant records a successful inner repair. The fixture is still deterministic and API-free, so it is a cheap gate for future repair-loop metrics before running expensive teacher benchmarks.
