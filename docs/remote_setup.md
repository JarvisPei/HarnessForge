# Remote API-Only Setup

This project should run experiments on a small always-on cloud VM even when both weak and teacher models are API models. The Mac is only for editing and lightweight checks.

## Source Of Truth

Keep code changes on the Mac working tree, push them to `main`, then pull `main` on the cloud VM before running experiments.

Do not treat files created or patched directly on the cloud VM as canonical source files. The cloud machine is for running experiments, capturing outputs, and debugging live runs. If you need to inspect or tweak behavior there, mirror the change back into the local repo before considering it part of the project.

## Why API-Only First

API-only iteration avoids the two slowest parts of early research:

- waiting for GPU allocation
- debugging local model serving

The first milestone is to validate the harness distillation loop:

```text
weak API run -> teacher API diagnosis -> harness patch -> regression
```

Only after this loop is stable should we add local weak models through vLLM or another chat-completions-compatible server.

## Server Setup

```bash
ssh <your-remote-host>
mkdir -p ~/projects ~/runs ~/tmp
cd ~/projects/<repo-name>
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

Run a named provider profile such as a custom suffix:

```bash
scripts/run_smoke.sh ALT_PROFILE
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

OpenAI-compatible relay APIs default to chat-completions-style behavior. Set `WEAK_API_STYLE=responses`, `TEACHER_API_STYLE=responses`, or profile-suffixed variants when a model/relay expects the OpenAI Responses API.

Provider defaults to `openai` for every profile. For Anthropic-native `/messages` behavior, set the relevant profile to:

```text
WEAK_PROVIDER_<PROFILE_NAME>=anthropic
TEACHER_PROVIDER_<PROFILE_NAME>=anthropic
```

For a relay that exposes Claude models through chat-completions-style endpoints, omit provider variables or set them to `openai`.

## Private Notes

Do not keep machine-specific runbooks or private credentials here if this repository is public. Put them in a separate private repo, a private gist, or an ignored local file under `docs/private/`.

### Public-safe rules

- keep `.env` out of git
- keep API keys out of git
- keep machine hostnames and internal paths out of public docs unless they are intentionally generic
- use `outputs/<run-id>/...` for experiment isolation instead of fresh clones
- use SSH remotes for code sync

### Recommended local pattern

```bash
git add <changed files>
git commit -m "<message>"
git push origin main
```

```bash
git pull --ff-only origin main
```

### Private runbook option

If you want to version a private remote runbook, keep it outside the public repository and load it locally from an ignored path, for example:

```text
docs/private/remote_setup.local.md
```

You can then add that directory to `.gitignore` while keeping the public `docs/remote_setup.md` clean.
