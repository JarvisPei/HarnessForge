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
ssh sandbox
cwork
git clone git@github.com:JarvisPei/AgentDistill.git
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

The CLI entrypoints call `load_dotenv(override=True)`, so you do not need to `source .env` manually before running experiments.

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

Use multiple iterations to measure actual impact:

```bash
APPLY_PATCHES=1 ITERATIONS=2 scripts/run_smoke.sh
```

Iteration 1 lets the teacher write a harness patch. Iteration 2 reloads the harness and runs the weak model again with the teacher's update included.

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

## Working Agreement

Keep this section current. It is the operational memory for long-running Codex sessions.

### Code Sync

Use `main` as the working branch unless the user explicitly asks for a branch.

The normal sync path is:

```text
Mac/local Codex edits -> push to GitHub main -> server pulls main
```

Current working rule set:

- keep one canonical server checkout at `/auxstore/cse/d11/data/s1155124388/projects/AgentDistill`
- keep `.env` on the server checkout and never commit or delete it during cleanup
- use `outputs/<run-id>/...` for experiment isolation instead of creating fresh repo clones
- clean only generated harness artifacts when needed; leave `.env`, `.venv`, and Git history intact
- treat `main` as the shared sync branch unless the user explicitly asks otherwise

On the Mac:

```bash
git add <changed files>
git commit -m "<message>"
git push origin main
```

On the server:

```bash
cd /auxstore/cse/d11/data/s1155124388/projects/AgentDistill
git pull --ff-only origin main
```

Do not create a new server clone for each experiment. The server should have one canonical checkout:

```text
/auxstore/cse/d11/data/s1155124388/projects/AgentDistill
```

### Server Environment

The server `.env` is private local state. It contains relay base URLs, API keys, and model names for weak/teacher profiles.

Rules:

- Never commit `.env`.
- Never delete `.env` as part of experiment cleanup.
- If the checkout is recreated, copy or recreate `.env` before running any API experiment.
- Prefer SSH Git remotes on the server: `git@github.com:JarvisPei/AgentDistill.git`.

### Experiment Isolation

Teacher runs can write generated harness files into `harness/` when patch application is enabled. Old generated harness files can contaminate later experiments, but creating a new clone for every experiment is not sustainable.

Use one canonical checkout and isolate experiments by resetting only generated experiment state before a new run:

```bash
cd /auxstore/cse/d11/data/s1155124388/projects/AgentDistill
git status --short
```

If the dirty files are only previous generated harness artifacts, clean the harness back to the Git baseline before the next experiment:

```bash
git restore harness
git clean -fd harness
```

This keeps `.env`, `.venv`, and Git history intact while removing old generated harness files. Do not run broader cleanup commands such as `rm -rf AgentDistill*` or `git clean -fdx` unless the user explicitly asks, because those can delete server-local state such as `.env` or cached environments.

Experiment outputs should be isolated with unique output directories or run IDs under `outputs/`, not with new repo clones.

### Terminal Discipline

After `ssh cse_H200`, run `ssh sandbox` before executing project commands. The login node should not run experiments.

Avoid accumulating unused terminals. Prefer one active SSH session for server work, and close or ignore stale sessions once the current run is complete.

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
