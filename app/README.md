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
