#!/usr/bin/env bash
# Full phase-1 run: download data, build the dataset, run walk-forward baselines.
# Usage: scripts/run_baselines.sh [extra stocklstm args]
# Set STOCKLSTM_DATA_DIR to keep data elsewhere (e.g. the TrueNAS NFS mount).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
. .venv/bin/activate
pip install -q -e .
stocklstm run "$@"
echo "Report: reports/baselines/metrics.md"
