# AgentDistill

AgentDistill is a small experimental scaffold for **harness distillation**: using a frontier teacher model to improve the external environment around a weaker model.

The first loop is intentionally simple:

1. Run the weak model on a task with the current harness.
2. Ask the teacher to diagnose the weak trace.
3. Ask the teacher for a concrete harness patch: prompt guideline, skill, tool idea, validator, state representation, or runtime policy.
4. Save the trace and patch for regression and later consolidation.

The goal is not to distill answers into the weak model. The goal is to distill frontier-agent behavior into a better runtime harness that the weak model can operate.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with your relay base URLs, keys, and model names. The API client uses OpenAI-compatible `/chat/completions` endpoints and does not require the official SDK.

## Run a Smoke Experiment

```bash
python -m agentdistill.run --config configs/smoke.yaml
```

Outputs are written under `outputs/`.

To use an alternate env suffix such as `*_CLAUDE`:

```bash
python -m agentdistill.run --config configs/smoke.yaml --profile CLAUDE
```

Provider defaults to OpenAI-compatible chat completions for every profile. Set `WEAK_PROVIDER_<PROFILE>=anthropic` and `TEACHER_PROVIDER_<PROFILE>=anthropic` only for native Anthropic `/messages` endpoints.

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

## Remote Server Workflow

The code is API-first and does not require GPUs for the smoke loop. This is the fastest path for early iteration: use API weak/teacher models, collect traces, then decide which harness updates are worth automating.

The H200 server becomes useful once we add local weak models, batch sweeps, or evaluation jobs.

Suggested remote workflow:

```bash
ssh cse_H200
squeue
```

When Duo asks for a device, choose `1`, then approve on your phone.

After login, clone or sync this project, create `.env` on the server, install dependencies, and run the same command:

```bash
python -m agentdistill.run --config configs/smoke.yaml
```

Do not commit `.env`.

### Suggested Phases

Phase 1: API-only loop on the remote server.

- weak model: API mini model
- teacher model: frontier API model
- goal: validate trace format, diagnosis quality, and harness patch taxonomy

Phase 2: local weak model served on H200.

- weak model: vLLM/OpenAI-compatible local endpoint
- teacher model: API frontier model
- goal: test whether harness evolution compensates for local model weaknesses

Phase 3: batched evaluation.

- run held-out suites
- compare prompt-only, skill, tool, validator, and runtime-policy updates
- measure teacher intervention cost and regression safety
