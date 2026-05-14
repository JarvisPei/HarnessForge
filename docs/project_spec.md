# Harness Distillation Project Spec

## Hypothesis

Small models can become substantially more capable as agents when a frontier model learns and maintains the harness around them.

The learning target is not the weak model's weights. The learning target is the external system:

- prompts
- skills
- tools
- validators
- state representations
- runtime policies
- generated helper code

## Core Loop

```text
task -> weak model + current harness -> trace
trace -> teacher diagnosis -> harness patch
harness patch -> regression test -> updated harness
```

The teacher should answer:

```text
What harness change would have made the weak model succeed?
```

not merely:

```text
What is the correct answer?
```

The implementation must preserve this boundary: humans should build the mechanism for applying teacher-generated patch bundles, but the concrete harness update should be produced by the teacher from traces. Manual edits to harness content are only for bootstrapping base structure or fixing infrastructure.

The minimal self-improvement experiment is:

```text
iteration 1: weak run -> teacher patch_bundle -> apply to harness
iteration 2: weak run with updated harness -> compare failure category and answer
```

Teacher-generated code is initially limited to `harness/tools/*.py`, must compile, and is exposed to the weak model as a tool specification. The framework should not execute arbitrary teacher code until there is a stronger sandbox and test gate.

## Update Types

### Prompt Guideline

Use for lightweight procedural reminders that the weak model can already follow.

Example: "Before finalizing arithmetic, recompute the expression in a second order."

### Skill

Use for reusable multi-step behavior.

Example: "Debugging functions with implicit initialization assumptions."

### Tool

Use when a repeated operation is too brittle for the weak model to perform directly.

Example: a deterministic unit-test runner, parser, webpage extractor, schema checker, or calculator.

### Validator

Use when the weak model can produce an answer but needs external checks before finalizing.

Example: verify that a code patch passes a generated minimal test.

### State Representation

Use when the weak model's context is too messy.

Example: compress a browser page into `{title, visible_controls, candidate_answers, evidence}` before asking for the next action.

### Runtime Policy

Use when the framework should make control-flow decisions around the weak model.

Example: retry on tool schema errors, force verification after code edits, or trigger teacher escalation only after repeated failed attempts.

## Initial Experiment Matrix

Baselines:

- weak model + static prompt
- weak model + SCOPE-style prompt evolution
- weak model + harness evolution
- teacher model + static prompt

Ablations:

- prompt only
- prompt + skills
- prompt + skills + tools
- prompt + skills + tools + validators/runtime

Metrics:

- task success rate
- tool error rate
- steps to success
- teacher intervention cost
- harness growth
- regression pass rate
- held-out generalization

## API-First Rationale

Using API models first keeps the loop simple and lets us validate the project mechanics before spending H200 queue time. The same scaffold can later swap the weak model client to a local vLLM endpoint on the server.

## H200 Use Cases

The remote H200 server becomes useful for:

- local weak model serving
- batched experiments
- held-out regression suites
- larger trace generation
- comparing local small models against API mini models
