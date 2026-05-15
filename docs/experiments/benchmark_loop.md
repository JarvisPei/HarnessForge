# Benchmarked Harness Evolution Loop

This workflow turns a single harness improvement case into a measurable loop.

## Command

```bash
python -m agentdistill.benchmark --config configs/benchmark_inventory.yaml
```

## Phases

1. `baseline_dev_probe`
   - Run teacher-visible dev probe tasks with the current harness.
   - Save weak answers, tool calls, and runtime policy results.
   - Do not request teacher diagnoses or apply patches.

2. `baseline_blind_test`
   - Run blind final-test tasks with the current harness.
   - Do not feed these results into teacher diagnoses.

3. `evolve_train_iter_XX`
   - Run train tasks.
   - Let the teacher apply patch bundles to the harness when failures occur.
   - Patches may create guidelines, skills, validators, tools, or runtime policies.
   - If the previous patch was rejected, feed a compact `patch_feedback` summary into `benchmark_context` so the teacher can repair the bundle.

4. `transfer_probe_iter_XX`
   - Run dev probe tasks after each train iteration.
   - Do not request teacher diagnoses or apply patches.
   - Feed the weak-model probe summary into the next train iteration as `benchmark_context`.

5. `after_dev_probe`
   - Rerun the dev probe tasks with the evolved harness.
   - Do not apply patches during this evaluation phase.

6. `after_blind_test`
   - Rerun blind final-test tasks with the evolved harness.
   - Do not feed these results into teacher diagnoses.

7. `dev_impact_report.json` and `blind_impact_report.json`
   - Compare before/after dev and blind results separately.
   - Track success, regression, improvement, answers, failure categories, and tool calls.

8. `metrics.json`
   - Summarize patch quality and transfer impact.
   - Count accepted/rejected patches, artifact types, contract failures, dev improvements, and blind improvements.

## Output Layout

```text
outputs/benchmark_inventory/<profile>/<run_id>/
  harness_before/
  harness_after/
  baseline_dev_probe/
  baseline_blind_test/
  evolve_train_iter_01/
  transfer_probe_iter_01/
  evolve_train_iter_02/
  transfer_probe_iter_02/
  after_dev_probe/
  after_blind_test/
  dev_impact_report.json
  blind_impact_report.json
  impact_report.json  # blind report alias for compatibility
  train_summary.json
  harness_files_after.json
  metrics.json
```

## Why This Matters

The benchmark runner separates harness learning from harness evaluation:

```text
train tasks: expose failures and let teacher modify harness
dev probe tasks: provide teacher-visible transfer feedback
blind test tasks: measure final generalization without teacher visibility
```

This is the first step toward reporting harness distillation as a repeatable experiment rather than a single anecdote.
