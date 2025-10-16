---
title: TravelBot Demo
emoji: "✈️"
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "docker"
app_file: streamlit_app.py
pinned: false
---

# Deploying TravelBot to Hugging Face Spaces (Docker)

<!-- The YAML above is required by Hugging Face Spaces when using the Docker template. -->

This file explains how to deploy this repository to a Hugging Face Space (Streamlit runtime).

1) Create a new Space
- Go to https://huggingface.co/spaces and create a new Space.
- Choose the Streamlit SDK and public/private as needed.

2) Repository and App file
- Point the Space to this repository (owner/repo) or push a copy of this repo into the new Space.
- In Space settings, set the _App file_ to `streamlit_app.py` (this wrapper runs `app/streamlit_app.py`).

3) Requirements
- The Space will install packages from `requirements.txt`. This repo includes `requirements.txt` at the root. Ensure it contains:
  - transformers
  - torch
  - streamlit
  - sentencepiece
  - evaluate
  - nltk
  - wordcloud
  - rouge-score
  - bert-score
  - safetensors

4) Model hosting
- If your model is public (https://huggingface.co/excelasaph/fine_tuned_t5_travel_geography): no token required; the model will be downloaded at runtime.
- If your model is private: create a secret in Space settings with key `HF_TOKEN` and value set to a valid Hugging Face token that has read access to the model repo.

5) Secrets and Environment
- Add a secret named `HF_TOKEN` in the Space Settings -> Secrets for private models.
- Optionally add `MODEL_ID` or edit the default in the app sidebar when running the Space.

6) Run & Debug
- Launch the Space. If model download fails, check the Space logs in the UI (Settings -> Logs) and ensure `HF_TOKEN` is present if required.
- The Streamlit UI will show an error message if the model cannot be loaded; logs in Spaces provide the stack trace.

7) Tips for faster startup
- Bundling the model in the repo drastically increases repo size (not recommended for large weights). Prefer Hub download.
- For a public model, first-run download will take time; subsequent runs in the same Space instance are faster.

8) Optional: recommended Space settings
- Increase CPU/RAM if your model requires more memory and your plan permits it.
- For private models, ensure the token has `read` scope.

If you'd like I can also add a small Health-check endpoint wrapper or a lightweight `requirements-locked.txt` for faster, reproducible installs.
# Deploying TravelBot to Hugging Face Spaces (Docker)

title: TravelBot Demo
emoji: "✈️"
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "docker"
app_file: streamlit_app.py
pinned: false

<!-- The YAML above is required by Hugging Face Spaces when using the Docker template. -->

This file explains how to deploy this repository to a Hugging Face Space (Streamlit runtime).

1) Create a new Space
- Go to https://huggingface.co/spaces and create a new Space.
- Choose the Streamlit SDK and public/private as needed.

2) Repository and App file
- Point the Space to this repository (owner/repo) or push a copy of this repo into the new Space.
- In Space settings, set the _App file_ to `streamlit_app.py` (this wrapper runs `app/streamlit_app.py`).

3) Requirements
- The Space will install packages from `requirements.txt`. This repo includes `requirements.txt` at the root. Ensure it contains:
  - transformers
  - torch
  - streamlit
  - sentencepiece
  - evaluate
  - nltk
  - wordcloud
  - rouge-score
  - bert-score
  - safetensors

4) Model hosting
- If your model is public (https://huggingface.co/excelasaph/fine_tuned_t5_travel_geography): no token required; the model will be downloaded at runtime.
- If your model is private: create a secret in Space settings with key `HF_TOKEN` and value set to a valid Hugging Face token that has read access to the model repo.

5) Secrets and Environment
- Add a secret named `HF_TOKEN` in the Space Settings -> Secrets for private models.
- Optionally add `MODEL_ID` or edit the default in the app sidebar when running the Space.

6) Run & Debug
- Launch the Space. If model download fails, check the Space logs in the UI (Settings -> Logs) and ensure `HF_TOKEN` is present if required.
- The Streamlit UI will show an error message if the model cannot be loaded; logs in Spaces provide the stack trace.

7) Tips for faster startup
- Bundling the model in the repo drastically increases repo size (not recommended for large weights). Prefer Hub download.
- For a public model, first-run download will take time; subsequent runs in the same Space instance are faster.

8) Optional: recommended Space settings
- Increase CPU/RAM if your model requires more memory and your plan permits it.
- For private models, ensure the token has `read` scope.

If you'd like I can also add a small Health-check endpoint wrapper or a lightweight `requirements-locked.txt` for faster, reproducible installs.

<!-- The YAML above is required by Hugging Face Spaces when using the Docker template. -->

This file explains how to deploy this repository to a Hugging Face Space (Streamlit runtime).

1) Create a new Space
- Go to https://huggingface.co/spaces and create a new Space.
- Choose the Streamlit SDK and public/private as needed.

2) Repository and App file
- Point the Space to this repository (owner/repo) or push a copy of this repo into the new Space.
- In Space settings, set the _App file_ to `streamlit_app.py` (this wrapper runs `app/streamlit_app.py`).

3) Requirements
- The Space will install packages from `requirements.txt`. This repo includes `requirements.txt` at the root. Ensure it contains:
  - transformers
  - torch
  - streamlit
  - sentencepiece
  - evaluate
  - nltk
  - wordcloud
  - rouge-score
  - bert-score
  - safetensors

4) Model hosting
- If your model is public (https://huggingface.co/excelasaph/fine_tuned_t5_travel_geography): no token required; the model will be downloaded at runtime.
- If your model is private: create a secret in Space settings with key `HF_TOKEN` and value set to a valid Hugging Face token that has read access to the model repo.

5) Secrets and Environment
- Add a secret named `HF_TOKEN` in the Space Settings -> Secrets for private models.
- Optionally add `MODEL_ID` or edit the default in the app sidebar when running the Space.

6) Run & Debug
- Launch the Space. If model download fails, check the Space logs in the UI (Settings -> Logs) and ensure `HF_TOKEN` is present if required.
- The Streamlit UI will show an error message if the model cannot be loaded; logs in Spaces provide the stack trace.

7) Tips for faster startup
- Bundling the model in the repo drastically increases repo size (not recommended for large weights). Prefer Hub download.
- For a public model, first-run download will take time; subsequent runs in the same Space instance are faster.

8) Optional: recommended Space settings
- Increase CPU/RAM if your model requires more memory and your plan permits it.
- For private models, ensure the token has `read` scope.

If you'd like I can also add a small Health-check endpoint wrapper or a lightweight `requirements-locked.txt` for faster, reproducible installs.