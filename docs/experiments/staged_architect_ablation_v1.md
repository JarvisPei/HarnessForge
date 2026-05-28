# Staged Architect Ablation

Date: 2026-05-28

This records the decision to keep staged architect mode as an experimental ablation and return the main research loop to one-pass teacher drafts plus focused repair.

## Context

The staged workflow split teacher work into:

```text
weak trace -> architect sketch -> harness developer bundle -> contract repair
```

The goal was to make executable harness generation more code-agent-like: first freeze scope, then generate tool/runtime-policy/tests, then let contract checks repair failures.

## Evidence

The staged path produced useful mechanism evidence:

- It generated accepted executable bundles.
- It exposed when a tool was accepted but had no activation path.
- Adding activation constraints led to accepted tool + runtime policy + test bundles.
- Inner repair successfully fixed a generated table parser contract failure.

However, the staged path also exposed a practical cost problem:

- Bundle generation became slow and hit relay `504 Gateway Time-out` once dev-style activation hints were added.
- The frozen sketch made narrow activation scopes easier to diagnose, but also made the system less flexible than the one-pass teacher.
- The additional teacher calls did not clearly improve transfer over the existing one-pass + repair loop.

## Decision

The mainline loop is:

```text
one-pass teacher draft -> contract-gated apply -> focused repair -> transfer probe
```

Staged mode remains available behind:

```bash
python -m agentdistill.benchmark --config <config> --architect-mode staged
```

but it should be treated as an ablation/experimental path, not the default project direction.

## Interpretation

The important research insight from staged mode is not that multi-round generation should replace one-pass. It is that executable harnesses need explicit activation and transfer contracts. Those contracts can be fed back into the one-pass teacher and repair loop without requiring a permanently multi-round workflow.

The current hypothesis is that a strong one-pass teacher draft, followed by compiler-like checks and focused repair feedback, is the better default engineering tradeoff.

