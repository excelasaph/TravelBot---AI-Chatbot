---
title: TravelBot Demo
emoji: "🚀"
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "docker"
app_file: app.py
pinned: false
short_description: TravelBot — Travel & Geography chatbot demo
license: mit
---

TravelBot — Hugging Face Space demo

This folder contains a minimal Streamlit app and supporting files to deploy a public demo of the TravelBot fine-tuned T5 model.

Files:
- `app.py` / `streamlit_app_space.py` — Streamlit app that loads a model from the Hub by repo id and provides a simple UI.
- `requirements.txt` — Python dependencies for the Space.
- `Dockerfile` — Dockerfile used by Spaces when choosing the Docker SDK/template.

Quick start (local / Streamlit deploy)

1) Create and activate a virtual environment (local test)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) Run the app locally

```powershell
streamlit run streamlit_app.py
```

Model hosting options

- Local bundle (current default): the app will attempt to load a local model folder. Place `fine_tuned_t5_travel_geography/` next to this `app/` folder before deploying if you want local-only inference.
- Hugging Face Hub download: to download from the Hub at runtime you must set an HF token in the host environment and adjust `load_model(..., local=False)` or implement an auto-fallback.
- Inference API / Endpoint: recommended for light-weight deploys. Store token in `st.secrets` or environment variables and call the API from the app.

Deployment notes

- Streamlit Community Cloud / Streamlit Deploy will use `requirements.txt` and run the app without a Dockerfile. If deploying with Render or Docker, ensure `requirements.txt` is installed during build and that model weights are available in the runtime or fetched securely.
- Avoid embedding secrets or tokens in the repo. Use platform secrets/ENV variables.

If you'd like, I can:
- Add an Inference-API fallback that uses `st.secrets["HF_TOKEN"]` when present, or
- Add a startup check that reports whether the local model folder is present and provides guidance.
