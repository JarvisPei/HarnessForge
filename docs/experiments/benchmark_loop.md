# Benchmarked Harness Evolution Loop

This workflow turns a single harness improvement case into a measurable loop.

## Command

```bash
python -m agentdistill.benchmark --config configs/benchmark_inventory.yaml
```

## Phases

1. `baseline_heldout`
   - Run held-out tasks with the current harness.
   - Save weak answers, teacher diagnoses, tool calls, and runtime policy results.

2. `evolve_train_iter_XX`
   - Run train tasks.
   - Let the teacher apply patch bundles to the harness when failures occur.
   - Patches may create guidelines, skills, validators, tools, or runtime policies.

3. `after_heldout`
   - Rerun the same held-out tasks with the evolved harness.
   - Do not apply patches during this evaluation phase.

4. `impact_report.json`
   - Compare before/after held-out results.
   - Track success, regression, improvement, answers, failure categories, and tool calls.

## Output Layout

```text
outputs/benchmark_inventory/<profile>/<run_id>/
  harness_before/
  harness_after/
  baseline_heldout/
  evolve_train_iter_01/
  evolve_train_iter_02/
  after_heldout/
  impact_report.json
  train_summary.json
  harness_files_after.json
```

## Why This Matters

The benchmark runner separates harness learning from harness evaluation:

```text
train tasks: expose failures and let teacher modify harness
held-out tasks: measure whether those modifications generalize
```

This is the first step toward reporting harness distillation as a repeatable experiment rather than a single anecdote.
