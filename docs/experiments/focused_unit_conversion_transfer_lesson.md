# Focused Unit Conversion Transfer Lesson

Date: 2026-05-22

This captures the main lesson from the focused unit-conversion family.

## What Changed

- The teacher repair loop worked better once transfer failures were prioritized instead of only feeding the first unresolved case.
- The final accepted harness was a generalized `measurement_arithmetic` tool plus a thin `force_measurement_arithmetic` runtime policy.
- The tool handled volume, mass, and length in one parser/executor path, while the policy stayed structural and did not hard-code answers.

## What We Learned

- The teacher should see the most informative failure first, especially tool failures, not just the first failed transfer case.
- Complex arithmetic and unit normalization belong in a deterministic tool, not in the router.
- A thin runtime policy is easier to generalize and easier to audit.

## Outcome

- Dev improved.
- Blind syrup improved instead of regressing.
- The server benchmark rerun completed with executable harness repairs.

## Operational Default

- Keep `REQUEST_TIMEOUT_SECONDS=600` as the baseline default for this workflow.

