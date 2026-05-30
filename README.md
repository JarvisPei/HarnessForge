# HarnessForge

HarnessForge is an open-source framework for **harness distillation**: using a frontier teacher model to improve the environment around a weaker model.

It does not try to distill answers into the weak model. It distills the surrounding system:

- prompts
- skills
- tools
- validators
- state representations
- runtime policies

The loop is intentionally simple:

1. Run the weak model on a task with the current harness.
2. Ask the teacher to diagnose the trace.
3. Ask the teacher for a concrete harness patch.
4. Apply contract-gated patches and repair rejected bundles.
5. Save the trace, tests, and patch for regression and later consolidation.

The working hypothesis is that small models become much more useful when the teacher keeps improving the harness they operate inside.

## Current Results

The project already has working end-to-end runs on the cloud workflow.

- inventory arithmetic exposed interface and transfer issues
- unit conversion showed that deterministic normalization can improve transfer
- structured extraction and validation improved both dev and blind probes with a normalization skill
- table-join aggregation improved dev and blind probes through a teacher-generated tool plus runtime policy

Latest structured extraction result:

- dev improved: 2
- blind improved: 2
- accepted harness update: normalization skill

Latest table-join runtime result:

- dev improved: 3
- blind improved: 2
- blind improvements with runtime effect: 2
- accepted harness update: deterministic table calculator + runtime policy + tests

The first compact evidence suite is documented in [docs/experiments/evidence_suite_v1.md](docs/experiments/evidence_suite_v1.md). The first boundary-case expansion is documented in [docs/experiments/evidence_suite_v2_boundary.md](docs/experiments/evidence_suite_v2_boundary.md).

## How It Works

```text
weak model -> trace -> teacher one-pass draft -> contract checks -> focused repair -> regression probes
```

The teacher is asked to change the harness, not to solve the task directly.

The current mainline is a one-pass teacher draft followed by focused repair feedback. A staged architect mode exists behind `--architect-mode staged` for ablations, but it is experimental and not the default path.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with your relay base URLs, keys, and model names. The current client uses chat-completions-style endpoints and does not require the official SDK.

For most demos, start with the default OpenAI GPT profile (`gpt-5.4-mini` weak / `gpt-5.5` teacher).

## Run a Smoke Experiment

```bash
python -m agentdistill.run --config configs/smoke.yaml
```

Outputs are written under `outputs/`.

To use a different profile or relay suffix:

```bash
python -m agentdistill.run --config configs/smoke.yaml --profile ALT_PROFILE
```

Provider defaults to the chat-completions path for every profile. Set `WEAK_PROVIDER_<PROFILE>=anthropic` and `TEACHER_PROVIDER_<PROFILE>=anthropic` only for native Anthropic `/messages` endpoints.

By default, teacher-suggested harness changes are stored as proposals under `outputs/.../patches`. To let the teacher update the harness files directly within the allowed `harness/` directories:

```bash
APPLY_PATCHES=1 python -m agentdistill.run --config configs/smoke.yaml
```

By default, only patches attached to diagnosed failures are applied. Add `--apply-success-patches` only when you intentionally want teacher-generated reinforcement patches from successful traces.

To let teacher changes affect the next weak-model attempt in the same run:

```bash
APPLY_PATCHES=1 ITERATIONS=2 scripts/run_smoke.sh
```

Tool-focused stress run:

```bash
python -m agentdistill.run --config configs/tool_stress.yaml --apply-patches --iterations 3
```

If a teacher-generated runtime policy exists, the runner evaluates it after the weak model's initial answer. A policy may force a tool call before the final answer is produced.

## Benchmark Families

The public benchmark map is intentionally small and composable. These families are the current public test surface for harness distillation; detailed run logs live under `docs/experiments/`.

- Inventory arithmetic: parser -> deterministic arithmetic tool -> runtime policy
- Unit conversion arithmetic: unit normalization tool + conversion-table tests + mixed-unit policy
- Structured extraction and validation: messy text -> strict JSON + normalization skill + schema-aware checks
- Table lookup and aggregation: table parser + filter/aggregate tool + validator
- Repair and transfer feedback: contract-gated repair loops that measure whether harness changes improve dev and blind probes

The planning notes in [docs/benchmark_family_plan.md](docs/benchmark_family_plan.md) track how these families expand over time.

To summarize completed benchmark runs as an evidence table:

```bash
python -m agentdistill.evidence --format markdown <run_dir> [<run_dir> ...]
```

## Remote Workflow

The code is API-first and does not require GPUs for the smoke loop. A single always-on cloud VM works well as the runtime control plane: API weak/teacher models, trace collection, harness updates, and isolated `outputs/` runs can all live there. The Mac stays for editing and lightweight checks.

Keep one canonical checkout on the remote machine. Do not create a fresh clone for every experiment. Keep `.env` private and use SSH Git remotes.

After login, sync the project, ensure `.env` exists, install dependencies, and run the smoke command:

```bash
python -m agentdistill.run --config configs/smoke.yaml
```

### Suggested Phases

Phase 1: API-only loop on the remote server.

- weak model: API mini model
- teacher model: frontier API model
- goal: validate trace format, diagnosis quality, and harness patch taxonomy

Phase 2: local weak model served on a stronger GPU machine.

- weak model: vLLM or other chat-completions-compatible local endpoint
- teacher model: API frontier model
- goal: test whether harness evolution compensates for local model weaknesses

Phase 3: batched evaluation.

- run held-out suites
- compare prompt-only, skill, tool, validator, and runtime-policy updates
- measure teacher intervention cost and regression safety

## Public / Private Split

This repository is meant to stay public-safe.

- public: code, configs, prompts, benchmark summaries, sanitized docs
- private: `.env`, machine-specific runbooks, API keys, and anything under `docs/private/`

Git cannot hide committed content in a public repository. If something must stay private, keep it outside the public repo and sync it with a separate private repository or an ignored local directory.

## Contributing

Issues and pull requests that improve benchmark families, harness contracts, or evaluation coverage are welcome.
