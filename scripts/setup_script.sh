#!/usr/bin/env bash
# setup venv
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Download base models and datasets
hf download shaundano/ai360-weld-data --repo-type dataset --max-workers 16
hf download shaundano/ai360-weld-models   --max-workers 16

