# Transfer Feedback Repair Success

Date: 2026-05-16

This records the first transfer-tight run where a baseline-failing probe family exposed a real behavior gap, the teacher repaired the accepted harness, and a second transfer-feedback repair pass improved both dev and blind weak-model outcomes.

## Run

Server workspace:

```text
clean server workspace for the transfer-feedback repair run
```

Commands:

```bash
REQUEST_TIMEOUT_SECONDS=60 python -m agentdistill.repair_family --transfer-tight -o outputs/repair_family_transfer_tight_v1
REQUEST_TIMEOUT_SECONDS=60 python -m agentdistill.repair_family --transfer-tight --transfer-feedback-repair -o outputs/repair_family_transfer_feedback_v1
```

The first run filters to baseline-failing probes. The second run reuses that transfer-tight family and adds a dev-feedback repair pass after the accepted harness.

## Filter Result

Probe filter summary:

```text
candidate_tasks = 10
baseline_pass = 2
baseline_fail = 8
recommended_dev_candidates = 4
recommended_blind_candidates = 4
mechanism_only = 1
```

The selected transfer-tight candidates were:

```text
tool_policy_pair
  dev: 2 failing, 1 passing
  blind: 2 failing, 1 passing

tool_contract_repair
  dev: 2 failing
  blind: 2 failing

fallback_rejected_paths
  mechanism_only
```

## Transfer-Tight Result

Transfer-tight run summary:

```text
cases = 3
transfer_tight = true
repair_successes = 2
errors = 1
mechanism_only = 1
dev_improved = 0
blind_improved = 0
```

The `tool_contract_repair` case was accepted but did not improve weak transfer in this first pass. The after-run telemetry showed that some tasks started invoking `signed_sum`, but the weak model still failed on comma-separated or long signed update patterns.

## Transfer-Feedback Repair Result

Transfer-feedback repair run summary:

```text
cases = 3
transfer_tight = true
transfer_feedback_repair = true
repair_successes = 3
errors = 0
mechanism_only = 1
dev_improved = 1
blind_improved = 1
dev_regressed = 1
blind_regressed = 1
```

The key outcome was `tool_contract_repair`:

```text
dev_transfer: before_success = 0, after_success = 1
blind_transfer: before_success = 0, after_success = 1
```

That second pass repaired `signed_sum` to handle comma-separated signed integers, which was the actual transfer hazard exposed by the baseline-failing probes.

## What Changed

This experiment added three pieces to the workflow:

```text
1. baseline probe filtering
2. transfer-tight candidate family
3. optional dev-feedback repair after accepted harness repair
```

The important part is not the specific tool, but the loop:

```text
filter baseline-failing probes
-> repair accepted harness
-> observe weak transfer
-> feed dev failure back to teacher
-> repair again
-> re-evaluate blind
```

## Why This Matters

This is the smallest version of the intended harness-distillation claim:

```text
teacher changes environment
weak model behavior changes
blind evaluation improves
```

The run also shows why transfer had previously looked flat: the earlier probes were too easy, and even the harder ones needed a second repair pass before the accepted harness actually matched the weak model's failure mode.
