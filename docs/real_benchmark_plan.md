# Real Benchmark Validation Plan

This document defines the next research phase for HarnessForge: moving from
synthetic mechanism probes to a real agent benchmark while preserving the core
teacher-as-architect hypothesis.

## Goal

HarnessForge should not become a collection of hand-tuned toy benchmarks. The
project goal is to test a general mechanism:

```text
weak trace -> frontier teacher diagnosis -> harness patch -> contract gate ->
repair -> held-out transfer
```

The weak model's weights stay fixed. The learned object is the harness around
the weak model: prompts, skills, tools, validators, state representations, and
runtime policies.

Synthetic benchmark families remain useful, but only as mechanism probes. The
next milestone is to show that the same loop improves a weak model inside a
public, stateful, tool-using agent benchmark.

## Why tau-bench

The first real benchmark target should be tau-bench / tau2-bench.

Reasons:

- It evaluates agents in stateful customer-service environments, not static QA.
- It includes domain policies, tools, databases, user simulation, and objective
  task success metrics.
- It has text-mode tasks that work with the current chat-completions model
  client.
- It exposes exactly the failures HarnessForge is meant to address: policy
  following, tool use, state updates, multi-turn recovery, and environment
  control.

Current public references:

- https://github.com/sierra-research/tau2-bench
- https://taubench.com/

As of the checked repository version, the text-mode starter domains include
official split metadata:

| domain | official splits |
| --- | --- |
| `airline` | `train: 30`, `test: 20`, `base: 50` |
| `retail` | `train: 74`, `test: 40`, `base: 114` |
| `telecom` | `small: 20`, `train: 74`, `test: 40`, `base: 114`, `full: 2285` |

`banking_knowledge` uses a different knowledge-task structure and should not be
the first integration target.

## What We Should Reuse

The benchmark should remain in charge of:

- domain task definitions
- user simulator
- official tools and databases
- environment state transitions
- success and policy evaluation

HarnessForge should replace or wrap only the agent-under-test side:

- weak model client
- agent prompt and response protocol
- teacher-generated skills
- teacher-generated tool wrappers or helper tools, after a safe execution path
  exists
- validators
- runtime policies
- state summaries passed to the weak model

For the first tau-bench adapter, teacher-generated helper tools are deferred.
Runtime policies may only select official tau-bench tools that the benchmark
environment can execute.

This boundary matters. If HarnessForge rewrites the benchmark environment or
leaks blind/test tasks to the teacher, the result stops being evidence of
harness distillation.

## Starting Scope

Start with text-mode `retail` or `airline`.

Recommended order:

1. `airline` smoke adapter: smaller task count, lower integration cost.
2. `retail` focused validation: richer policy and tool-use surface.
3. `retail` full text-mode run after the focused validation is stable.
4. `telecom` only after the method works on the simpler text domains.
5. `banking_knowledge` only when we intentionally study retrieval-aware
   harnesses.

Do not start with voice/full-duplex evaluation. That would add audio and
realtime-system variables before the text harness mechanism is validated.

## Evaluation Protocol

Use the official benchmark split before creating any custom split. The default
mapping should be:

```text
official train -> HarnessForge probe/dev/repair
official test  -> blind final evaluation
```

Within official `train`, use three non-overlapping task roles.

```text
probe/dev tasks:
  visible to weak traces and teacher diagnosis

repair/regression tasks:
  used by contract gates and focused repair

blind/test tasks:
  drawn from the official test split and never visible to the teacher during
  harness construction
```

The teacher may see weak-model traces from probe/dev tasks. It may generate
harness patches and repair them against allowed regression cases. It must not
see blind/test answers, traces, or failure labels before final evaluation.

Once an official `test` task has influenced a teacher prompt, human decision, or
harness repair, it is no longer a test task for that run. Use official `test`
sparingly and report every evaluation pass.

For each benchmark slice, run:

1. Weak baseline with the default HarnessForge agent wrapper.
2. Weak plus static prompt baseline.
3. Weak plus teacher-evolved harness.
4. Optional frontier direct baseline for context, not as the target system.

The main comparison is:

```text
weak baseline vs weak + teacher-evolved harness
```

## Success Criteria

A tau-bench run counts as useful evidence only if it satisfies all of these:

- blind task success improves over the weak baseline
- blind regressions are low or explainable
- accepted harness artifacts are produced by the teacher, not hand-authored
- runtime changes can be observed when the accepted artifact is a tool or policy
- the same harness survives at least one held-out slice with surface variation
- teacher cost and number of repair rounds are recorded

A negative result is still useful if it identifies a reusable failure mode, such
as tool routing failure, policy overreach, brittle state summaries, or missing
semantic coverage.

## Ablations

The first real-benchmark ablation set should stay small:

- prompt-only
- skill-only or guideline-only
- tool and validator enabled
- runtime policy enabled
- full HarnessForge loop

Avoid a large ablation grid until the adapter produces one stable positive or
diagnostic negative result.

## Metrics

Record benchmark metrics:

- task success rate
- policy violation rate
- number of turns
- tool call success and error rate
- final environment-state correctness

Record HarnessForge metrics:

- accepted and rejected patch counts
- artifact types in accepted bundles
- repair rounds per accepted bundle
- teacher tokens, latency, and retry count
- weak-model tokens and latency
- blind improvements and regressions
- whether blind wins coincide with runtime effects

## Implementation Plan

### Step 1: Adapter Survey

Read the tau2-bench agent API and identify the smallest stable integration
point for replacing the agent-under-test. Do not modify tau2-bench itself unless
the adapter cannot be built otherwise.

Deliverable:

```text
docs/experiments/tau_bench_adapter_notes.md
```

This survey is complete for the first adapter design. The recommended
integration point is a custom half-duplex agent registered through the
tau2-bench registry and run through the programmatic runner API.

### Step 2: Text Smoke

Run a tiny text-mode slice with the weak model through the official tau2-bench
environment. No teacher harness evolution yet.

Deliverable:

```text
weak baseline traces + task outcomes
```

### Step 3: Harness Patch Integration

Map tau-bench traces into the existing HarnessForge diagnosis schema. The
teacher should receive enough context to act as an architect:

- task instruction
- visible user/agent turns
- tool calls and results
- policy snippets available to the agent
- final success/failure signal on probe/dev tasks

Deliverable:

```text
one accepted teacher-generated harness patch that affects a later tau-bench run
```

### Step 4: Focused Transfer Run

Run a small split:

```text
dev: 5 to 10 tasks
blind: 5 to 10 tasks
```

The goal is not leaderboard performance. The goal is to show measurable transfer
from teacher-generated harness changes under strict blind separation.

Deliverable:

```text
evidence table with weak baseline vs evolved harness
```

### Step 5: Full Domain Run

After a focused positive or clear diagnostic negative result, scale to full
`airline` or `retail` text-mode.

Deliverable:

```text
docs/experiments/tau_bench_text_v1.md
```

## Non-Goals For This Phase

- Do not tune against the full blind/test set.
- Do not use same-teacher self-audit as a substitute for held-out evaluation.
- Do not chase telecom, banking knowledge, voice, WebArena, or SWE-bench before
  the tau-bench text adapter is validated.
- Do not treat synthetic benchmark success as the final project proof.

## Decision Point

After the first focused tau-bench transfer run, decide between:

- continue scaling tau-bench because harness changes transfer,
- improve the adapter because runtime effects are not reaching the weak model,
- revise the teacher contract because generated artifacts are too brittle, or
- switch benchmark only if tau-bench is structurally mismatched to the method.

The default assumption is that tau-bench is the right first real validation
target. The next work should build the adapter, not add another synthetic
benchmark family.
