# Tau Cancel Progress Controller v1

Date: 2026-06-15

## Claim

A teacher-generated runtime progress controller changed weak-model behavior on a real tau-bench airline cancellation task. The effect is visible at the tool-action level, not only in final text.

## Setup

Task slice: airline train tasks `0, 1, 39, 41, 47, 49`.

Baseline disables generated harness directories:

```text
outputs/tau_bench_ablation/cancellation_family_weak_only_empty_harness_v1
```

Teacher-harness run uses the generated cancellation progress controller:

```text
outputs/tau_bench_policy/cancellation_family_after_eligible_progress_v1
```

## Key Result

Family aggregate:

```text
weak-only empty harness: official_passed 2/6, matched_write_actions 1/3
teacher harness:          official_passed 5/6, matched_write_actions 3/3
```

Task 39 is the cleanest comparison:

```text
weak-only:       reward 0.0, matched writes 1/3, actual cancel ids [8C8K4E]
teacher harness: reward 1.0, matched writes 3/3, actual cancel ids [8C8K4E, LU15PA, MSJ4OA]
```

The weak-only agent either stopped early or leaked malformed tool-call text. The teacher-generated policy reconstructed progress state and forced the remaining eligible `cancel_reservation` calls after user confirmation.

## Limitations

- Weak-only tasks `0`, `41`, and `49` ended with adapter/user-simulator empty-message errors.
- The teacher-harness family run timed out on task `41`.
- A separate task `41` rerun passed official reward but still had one strict action mismatch.

So this is evidence of a real task-level harness effect and a positive family-level signal, not yet a fully clean transfer proof.

## Next Action

Stabilize tau evaluation noise before promoting this harness pattern: handle empty user-simulator messages, keep strict action reporting, and rerun the same fixed family before expanding the benchmark slice.
