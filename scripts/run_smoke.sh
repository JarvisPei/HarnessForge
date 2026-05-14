#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  echo "Missing .env. Copy .env.example to .env and fill in API settings." >&2
  exit 1
fi

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

PROFILE="${1:-}"
APPLY_PATCHES="${APPLY_PATCHES:-0}"
ITERATIONS="${ITERATIONS:-1}"
EXTRA_ARGS=()
if [[ "$APPLY_PATCHES" == "1" ]]; then
  EXTRA_ARGS+=(--apply-patches)
fi
EXTRA_ARGS+=(--iterations "$ITERATIONS")

if [[ -n "$PROFILE" ]]; then
  python -m agentdistill.run --config configs/smoke.yaml --profile "$PROFILE" "${EXTRA_ARGS[@]}"
else
  python -m agentdistill.run --config configs/smoke.yaml "${EXTRA_ARGS[@]}"
fi
