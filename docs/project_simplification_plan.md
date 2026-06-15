# Project Simplification Plan

This note applies a five-step simplification lens to HarnessForge. Its purpose is to keep the project focused while we test whether teacher-generated harnesses transfer beyond single-task repairs.

## 1. Question Requirements

The core requirement is:

> A frontier teacher can inspect weak-model failure traces and generate harness changes that make weak+harness outperform weak-only on held-out workflow tasks, without worsening strict action coverage or write-tool safety.

Everything else is optional until it helps prove this claim.

Current non-requirements:

- The harness does not need to be fully domain-general.
- The teacher does not need a multi-agent or staged workflow by default.
- Generated harness artifacts do not need to be committed to the public repo.
- Metadata does not need to describe every implementation detail.
- More benchmark families are not automatically better unless they clarify transfer or regression risk.

## 2. Delete or Disable

Default path to keep:

- one-pass teacher draft
- focused repair loop
- contract validation
- dev and held-out family evaluation
- strict tau reporting

Default path to disable:

- staged teacher generation
- critic agents
- large metadata schemas
- generated harness commits
- benchmark expansion without a transfer question

These mechanisms can still be used for ablations, but they should not be part of the mainline unless a trace/report shows they remove a real blocker.

## 3. Simplify

The main lifecycle should stay:

```text
weak baseline
-> teacher diagnosis
-> teacher-generated harness
-> contract validation
-> dev repair task
-> held-out family transfer
-> strict report
```

Harness artifacts should stay in three operational classes:

- `skill` / `guideline`: visible to the weak model
- `runtime_policy` / `validator`: automatic runtime control
- `tool` / code: new deterministic environment capability

For generated harness metadata, use the smallest useful registry index:

```json
{
  "name": "tau_airline_cancel_eligible_progress",
  "artifact_type": "runtime_policy",
  "domain": "tau_airline",
  "workflow": "cancel_all_upcoming",
  "risk": "destructive_write",
  "status": "candidate"
}
```

Precise trigger logic should live in executable harness code and tests, not in a new metadata DSL.

## 4. Accelerate

Add only mechanisms that reduce experiment cost or prevent misclassification.

High-value acceleration:

- automatic weak-only vs weak+harness reports
- timeout/API failure labels separate from behavioral failures
- strict action coverage next to official reward
- compact teacher context built from failed traces and active harness summaries
- fixed small family runs before large benchmark sweeps

Low-value acceleration for now:

- automatically generating many harnesses
- complex harness selectors
- frontier critic calls on every patch
- broad benchmark sweeps before a stable ablation protocol exists

## 5. Automate Last

Automate evaluation before automating generation.

Near-term automation:

- run a fixed tau family
- summarize official reward, strict action coverage, write-tool calls, and timeouts
- compare weak-only with weak+harness
- produce compact teacher input for the next repair

Defer:

- autonomous architect swarms
- always-on critic agents
- global harness retrieval systems
- multi-domain policy registries

## Add-Mechanism Gate

Before adding a new mechanism, answer all four questions:

1. Which observed trace/report failure does this solve?
2. Does it replace manual judgment or reduce experiment time?
3. Can it be disabled by default or scoped to one lifecycle step?
4. How will we know it improves transfer rather than just fixing one task?

If the answer is unclear, do not add the mechanism yet.

## Current Next Milestone

Use tau-bench text-mode as the real benchmark thread:

1. Establish weak-only baselines on a small fixed airline family.
2. Run weak+teacher-generated harness on the same family.
3. Compare official reward, strict action coverage, write-tool calls, and timeout/API failures.
4. Promote a harness from `candidate` only if it improves at least one dev task and does not regress held-out family behavior under the strict report.

This keeps the project aimed at harness transfer instead of accumulating untested infrastructure.
