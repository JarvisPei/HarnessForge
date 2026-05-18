# Runtime Policy Transfer Feedback Regression

Date: 2026-05-18

This records the narrow regression gate that came out of the `tool_contract_repair` transfer-feedback repair path.

## What It Protects

The key signal is:

```text
transfer_feedback.failure_mode = policy_or_routing_failure
transfer_feedback.recommended_repair_target = runtime_policy
transfer_feedback.repair_plan.primary_axis = runtime_policy
```

When that appears, the repair loop should route the teacher to the runtime-policy/test layer, not a tool-only repair.

## Evidence

Server run:

```text
repair_family --transfer-tight --transfer-feedback-repair
```

Relevant case:

```text
tool_contract_repair
```

The transfer-feedback repair substage accepted a runtime-policy bundle and improved blind transfer on the held-out runtime-policy probe family.

Observed substage result:

```text
transfer_feedback_repair.attempted = true
transfer_feedback_repair.repair_success = true
dev_transfer.improved = 0
blind_transfer.improved = 1
```

## Regression Checks

Use this as the fixed gate for future edits:

```text
1. build_transfer_feedback must classify the POSTED-updates failure as runtime_policy.
2. _repair_scope_for_transfer_feedback must route that repair_plan axis to runtime_policy/test paths.
3. The repair-family transfer-feedback path must remain able to produce an accepted runtime-policy bundle.
```

## Interpretation

This is not a global benchmark claim. It is a regression target for one repair mechanism:

```text
teacher sees transfer failure
-> repair_plan says runtime_policy
-> teacher repairs policy/test layer
-> blind transfer improves
```
