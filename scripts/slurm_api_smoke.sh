#!/usr/bin/env bash
# Optional Slurm wrapper for API-only runs. This should request CPU resources,
# not H200 GPUs. Cluster partition names vary, so adjust SBATCH lines as needed.

#SBATCH --job-name=agentdistill-api-smoke
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT_DIR/outputs/slurm"
cd "$ROOT_DIR"

source .venv/bin/activate
python -m agentdistill.run --config configs/smoke.yaml
