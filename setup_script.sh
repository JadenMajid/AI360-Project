# setup venv
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Download base models and datasets
hf download --repo-type dataset shaundano/ai360-weld-data
hf download shaundano/ai360-weld-models

