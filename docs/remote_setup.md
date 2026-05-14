# Remote API-Only Setup

This project should run experiments on the server even when both weak and teacher models are API models. The Mac is only for editing and lightweight checks.

## Why API-Only First

API-only iteration avoids the two slowest parts of early research:

- waiting for GPU allocation
- debugging local model serving

The first milestone is to validate the harness distillation loop:

```text
weak API run -> teacher API diagnosis -> harness patch -> regression
```

Only after this loop is stable should we add local weak models through vLLM or another OpenAI-compatible server.

## Server Setup

```bash
ssh cse_H200
git clone https://github.com/JarvisPei/AgentDistill.git
cd AgentDistill
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` on the server and fill in the relay endpoints:

```bash
nano .env
```

Run:

```bash
scripts/run_smoke.sh
```

Run a named provider profile such as `*_CLAUDE`:

```bash
scripts/run_smoke.sh CLAUDE
```

By default, teacher harness patches are proposals only. To let the teacher write patch bundles into the allowed `harness/` directories:

```bash
APPLY_PATCHES=1 scripts/run_smoke.sh
```

Apply mode only applies patches from traces with non-empty `failure_categories` by default. This keeps successful examples from bloating the harness with optional reinforcements.

## Environment Variables

The weak and teacher roles are independent so they can point to different providers or relay routes:

```text
WEAK_BASE_URL=
WEAK_API_KEY=
WEAK_MODEL=

TEACHER_BASE_URL=
TEACHER_API_KEY=
TEACHER_MODEL=
```

For relay APIs, the only requirement is OpenAI-compatible `/chat/completions` behavior.

Provider defaults to `openai` for every profile. For Anthropic-native `/messages` behavior, set:

```text
WEAK_PROVIDER_CLAUDE=anthropic
TEACHER_PROVIDER_CLAUDE=anthropic
```

For a relay that exposes Claude models through OpenAI-compatible chat completions, omit provider variables or set them to `openai`.

## Codex On Server

Running Codex directly on the server can be useful later, but it adds setup cost:

- install/configure Codex on the server
- provide a custom API base URL and key
- handle auth/session state on the remote machine
- ensure generated edits and experiment runs happen in the same repo checkout

Recommended order:

1. Use local Codex to edit code and SSH into the server for commands.
2. Stabilize the API-only experiment loop.
3. Add server-side Codex only if remote iteration becomes bottlenecked by SSH command orchestration.

This keeps the first experimental loop simple and reproducible.

## Later: Local Weak Model

Once API-only harness evolution works, serve a local weak model on H200 with an OpenAI-compatible endpoint and only change:

```text
WEAK_BASE_URL=http://localhost:<port>/v1
WEAK_MODEL=<local-model-name>
```

The rest of the harness loop should stay unchanged.
