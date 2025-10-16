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

Quick steps (PowerShell)
1) Activate your venv
```powershell
.\.venv\Scripts\Activate.ps1
```

2) (Optional) Install dependencies locally
```powershell
pip install -r requirements.txt
```

3) Create a model repo on Hugging Face (web UI recommended) and upload your `fine_tuned_t5_travel_geography` folder (see project `upload_model.py` helper or use git LFS).

4) Create a new Space on Hugging Face (https://huggingface.co/spaces) and choose "Docker" as the SDK (select the Streamlit Docker template in the UI). Make it Public for free hosting.

5) Push this `hf_space_demo` folder to your new Space repo. Example commands (replace `YOUR_SPACE_REPO`):
```powershell
cd hf_space_demo
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git add .
git commit -m "Add Streamlit demo"
git push origin main
```

6) In the Space UI, set the `MODEL_ID` in the sidebar or hardcode it into `streamlit_app_space.py`.

Notes & tips:
- For large models: consider quantization (bitsandbytes) and testing locally first.
- Public Space + public model = free demo. Private resources are paid.
- If the model is too big for the free Space (OOM/timeouts), consider using a smaller model for the public demo or a paid Inference Endpoint for production.

If you'd like, I can:
- add a robust Inference-API fallback to `streamlit_app_space.py` (requires a Space secret token), or
- add a quantization helper to reduce model size and memory requirements.

Which should I do next?
