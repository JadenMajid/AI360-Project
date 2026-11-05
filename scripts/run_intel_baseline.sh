#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${INTEL_BASELINE_PYTHON:-.venv/bin/python}"
CONFIG="${INTEL_BASELINE_CONFIG:-configs/intel_baseline.json}"
DATA_ROOT="${INTEL_BASELINE_DATA_ROOT:-data/intel_baseline}"
OUTPUT_DIR="${INTEL_BASELINE_OUTPUT_DIR:-artifacts/intel_baseline}"
DEVICE="${INTEL_BASELINE_DEVICE:-cuda}"
COMMON=(--config "$CONFIG" --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR" --device "$DEVICE")
mkdir -p "$OUTPUT_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ -f "$DATA_ROOT/audit.json" ]]; then
    "$PYTHON" -m src.intel_baseline prepare "${COMMON[@]}" --audit-only
else
    "$PYTHON" -m src.intel_baseline prepare "${COMMON[@]}"
fi
"$PYTHON" -m src.intel_baseline sweep "${COMMON[@]}" --resume
"$PYTHON" -m src.intel_baseline evaluate "${COMMON[@]}"
"$PYTHON" -m src.intel_baseline report "${COMMON[@]}"
