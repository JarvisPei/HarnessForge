# Tool Runtime Success Case

Date: 2026-05-14

This experiment is the first end-to-end success case for teacher-generated harness changes that go beyond prompt edits.

## Setup

Command:

```bash
python -m agentdistill.run --config configs/tool_stress.yaml --apply-patches --iterations 3
```

Weak model:

```text
gpt-5.4-mini
```

Teacher model:

```text
gpt-5.5
```

Task:

```text
A warehouse had 1,204 labels. It printed 37 sheets with 18 labels each, discarded 89 damaged labels, shipped 647 labels, then received 23 packs with 14 labels each. How many labels remain?
```

Expected computation:

```text
1204 + 37*18 - 89 - 647 + 23*14 = 1456
```

## What Happened

### Iteration 1

The weak model answered with the wrong total:

```text
1136 labels remain.
```

The teacher generated and applied a Python tool:

```text
harness/tools/inventory_arithmetic.py
```

This was not a manual fix. The teacher wrote the tool as a patch bundle.

### Iteration 2

The weak model still did not produce a clean tool call. The teacher diagnosed that the issue was no longer missing arithmetic guidance, but missing control-flow enforcement.

The teacher generated and applied a runtime policy:

```text
harness/runtime_policies/force_inventory_arithmetic_tool.py
```

The runtime policy forced use of the teacher-generated tool when the weak model failed to call it cleanly.

Observed tool result:

```json
{
  "ok": true,
  "start": 1204,
  "total_additions": 988,
  "total_subtractions": 736,
  "total": 1456,
  "expression_summary": "1204 + 988 - 736 = 1456"
}
```

### Iteration 3

The weak model produced a clean JSON tool call:

```json
{
  "tool_call": {
    "name": "inventory_arithmetic",
    "input": {
      "start": 1204,
      "additions": [
        {"count": 37, "per": 18, "label": "printed labels"},
        {"count": 23, "per": 14, "label": "received labels"}
      ],
      "subtractions": [
        {"count": 89, "label": "discarded damaged labels"},
        {"count": 647, "label": "shipped labels"}
      ]
    }
  }
}
```

The framework executed the tool and sent the result back to the weak model. The final answer was correct:

```text
1,456 labels remain. The printed and received labels were added, then the discarded and shipped labels were subtracted.
```

The teacher diagnosis for iteration 3 had no failure categories.

## Why This Matters

This validates the core harness distillation loop:

```text
weak failure
-> teacher diagnosis
-> teacher-generated tool code
-> weak still fails to use tool
-> teacher-generated runtime policy
-> framework changes control flow
-> weak succeeds with tool result
```

The key result is that the teacher did not merely add instructions. It changed the agent's executable environment.

## Current Limitations

- The task is still toy-scale and hand-authored.
- The runtime policy was generated in response to one narrow inventory arithmetic failure.
- Teacher-generated harness artifacts are not yet versioned as first-class experiment outputs.
- There is no automatic before/after metric aggregation beyond saved traces and summaries.
- Runtime policy code is statically restricted but not executed inside a hardened sandbox.

## Next Experimental Need

The next step is to turn this from a single success case into a measurable loop:

```text
benchmark set -> baseline weak run -> teacher harness evolution -> held-out rerun -> regression report
```

The system needs a harness-version store, impact evaluator, and held-out task suite to measure whether teacher-generated tools and policies generalize beyond the triggering example.
