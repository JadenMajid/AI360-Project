#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON="${INTEL_BASELINE_PYTHON:-.venv/bin/python}"
CONFIG="configs/intel_muon_once.json"
DATA_ROOT="data/intel_baseline"
OUTPUT_DIR="artifacts/intel_muon_once"
COMMON=(--config "$CONFIG" --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR" --device cuda)
mkdir -p "$OUTPUT_DIR"
if [[ ! -e "$OUTPUT_DIR/cache" ]]; then
    ln -s ../intel_baseline/cache "$OUTPUT_DIR/cache"
fi
"$PYTHON" -m src.intel_baseline sweep "${COMMON[@]}" --resume
"$PYTHON" -m src.intel_baseline evaluate "${COMMON[@]}"
