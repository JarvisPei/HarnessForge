# Inventory Benchmark Result

Date: 2026-05-14

This records the first benchmarked harness evolution run from a clean checkout.

## Command

```bash
REQUEST_TIMEOUT_SECONDS=300 python -m agentdistill.benchmark \
  --config configs/benchmark_inventory.yaml \
  --run-id clean_inventory_v1
```

## Result

The benchmark loop completed end to end:

```text
baseline_heldout -> evolve_train_iter_01 -> evolve_train_iter_02 -> after_heldout -> impact_report.json
```

The train phase produced teacher-generated harness code:

```text
harness/tools/inventory_arithmetic.py
harness/runtime_policies/force_inventory_arithmetic.py
```

Held-out impact:

```text
improved = 0
regressed = 0
```

## What Failed

The benchmark correctly exposed a generalization failure.

In the clean run, the teacher-generated runtime policy forced the `inventory_arithmetic` tool, but it passed a raw `task_instruction` field instead of the structured input schema expected by the tool.

Observed after-run tool call shape:

```json
{
  "name": "inventory_arithmetic",
  "input": {
    "task_instruction": "..."
  }
}
```

But the generated tool expected structured input similar to:

```json
{
  "start": 2050,
  "additions": [
    {"count": 16, "per": 45, "label": "printed tags"},
    {"count": 9, "per": 32, "label": "received tags"}
  ],
  "subtractions": [
    {"count": 138, "label": "misprinted tags"},
    {"count": 1107, "label": "sold tags"}
  ]
}
```

As a result, the after-run still failed held-out tasks even though the framework had successfully applied teacher-generated tool and runtime policy patches.

## Why This Is Useful

This is the first successful negative result from the benchmark loop:

```text
teacher can generate executable harness code
but generated components need interface compatibility checks
```

The next infrastructure gap is not more prompting. It is a contract checker between generated runtime policies and generated tools.

## Next Step

Add harness contract validation:

```text
tool schema/spec -> runtime policy proposed tool_input -> contract test -> accept/reject patch
```

The framework should run teacher-generated regression tests against generated tools and policies before accepting them into the evolved harness.
