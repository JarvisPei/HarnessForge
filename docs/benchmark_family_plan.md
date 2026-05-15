# Benchmark Family Expansion Plan

The next experiments should test whether teacher-generated harness distillation transfers beyond inventory arithmetic.

## Selection Criteria

Each benchmark family should have:

- train tasks that expose a repeatable weak-model failure mode
- held-out tasks with the same latent structure but different surface wording
- a plausible harness improvement beyond prompt text
- an objective checker based on expected answers or structured outputs
- room for teacher-generated tools, tests, and runtime policies

## Candidate Families

### Inventory Arithmetic

Status: first positive result.

Harness target:

```text
text parser -> deterministic arithmetic tool -> runtime policy
```

Purpose:

```text
Verify the core loop with tool/test/policy bundles and held-out transfer.
```

### Unit Conversion Arithmetic

Example failure:

```text
Convert mixed units, apply arithmetic, and return one normalized unit.
```

Harness target:

```text
unit normalization tool + conversion table tests + policy for mixed-unit prompts
```

Why useful:

```text
Tests whether generated tools can encode domain tables rather than only arithmetic parsing.
```

### Table Lookup And Aggregation

Example failure:

```text
Given a small markdown/CSV-style table, filter rows and aggregate a column.
```

Harness target:

```text
table parser + aggregation tool + validator for row/column references
```

Why useful:

```text
Exercises state representation and structured extraction, not just calculator use.
```

### Structured Extraction And Validation

Example failure:

```text
Extract fields from messy text into strict JSON and reject missing/contradictory fields.
```

Harness target:

```text
schema validator + normalization skill + runtime policy that rejects malformed output
```

Why useful:

```text
Tests harness improvements that constrain output shape rather than solve numeric tasks.
```

### Small Code Debugging

Example failure:

```text
Diagnose a short buggy function and produce a corrected implementation.
```

Harness target:

```text
unit-test generator + execution tool + policy that requires running tests before final answer
```

Why useful:

```text
Moves from deterministic parsing into code execution workflows while preserving objective tests.
```

## Metrics To Compare Across Families

For each run, compare:

- held-out before/after success
- improved and regressed counts
- accepted and rejected patch counts
- accepted bundle composition: tool, test, runtime policy, guideline, skill, validator
- contract failures caught before harness mutation
- final harness file count by artifact type
- whether the teacher modifies existing reusable components or creates near-duplicates

## Near-Term Recommendation

Add one family at a time. The next family should be unit conversion arithmetic because it is close enough to inventory arithmetic to reuse the loop, but different enough to require a new tool vocabulary and conversion tests.
