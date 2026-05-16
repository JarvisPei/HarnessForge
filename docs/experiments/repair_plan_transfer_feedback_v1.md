# Repair Plan Transfer Feedback v1

Date: 2026-05-16

This records the first step from a single `recommended_repair_target` toward a structured transfer-feedback repair plan.

## Change

Transfer feedback now keeps the existing `failure_mode` and `recommended_repair_target`, and also adds:

```json
{
  "repair_plan": {
    "primary_axis": "tool | runtime_policy | finalization",
    "allowed_artifact_types": ["..."],
    "required_regression_test": "..."
  }
}
```

The teacher prompt instructs the frontier teacher to follow `repair_plan.primary_axis`, `allowed_artifact_types`, and `required_regression_test` before writing patch bundles, while still checking the concrete failure evidence.

## Verification

Local tests:

```text
.venv/bin/python -m pytest tests/test_runtime_policy.py -q
77 passed
```

Direct server probe:

```text
runtime_policy target -> patch_type=runtime_policy
tool target           -> patch_type=tool
finalization target   -> patch_type=prompt_guideline
```

Lightweight server probe:

```bash
REQUEST_TIMEOUT_SECONDS=120 TEACHER_TIMEOUT_SECONDS=600 WEAK_TIMEOUT_SECONDS=120 \
  ../AgentDistill_bench_clean/.venv/bin/python -m agentdistill.repair_family \
  --output-dir outputs/repair_family_transfer_feedback_plan_v2 \
  --transfer-tight \
  --transfer-feedback-repair
```

Overall summary:

```text
repair_successes = 2
dev_improved = 1
blind_improved = 0
dev_regressed = 0
blind_regressed = 1
```

The full benchmark-level summary still has one blind regression from the outer `tool_policy_pair` case. The transfer-feedback repair substage, where `repair_plan` was actually present, was cleaner:

```text
transfer_feedback_repair.attempted = true
transfer_feedback_repair.repair_success = true
transfer_feedback_repair.dev_transfer:   improved=2, regressed=0
transfer_feedback_repair.blind_transfer: improved=1, regressed=0
```

The transfer feedback file included:

```text
failure_mode = tool_failure
recommended_repair_target = tool
repair_plan.primary_axis = tool
repair_plan.allowed_artifact_types = ["tool", "test"]
```

## Interpretation

The direct probe confirms the teacher can follow the structured repair axis across the three target types. The lightweight repair-family probe shows the transfer-feedback substage remained focused and improved dev/blind without regression. The outer-case regression means this is not yet a global benchmark win; it is a positive mechanism signal for the transfer-feedback repair path specifically.
