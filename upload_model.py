"""
Simple helper to upload a local model folder to Hugging Face Hub using huggingface_hub.upload_folder
Usage:
  1. Set REPO_ID to your repo (e.g. username/modelname)
  2. Run: python upload_model.py

Note: run `huggingface-cli login` in PowerShell first to authenticate.
"""
from huggingface_hub import upload_folder

REPO_ID = "excelasaph/fine_tuned_t5_travel_geography"  # e.g. excelasaph/fine_tuned_t5_travel_geography
LOCAL_FOLDER = r"c:\Users\Excel\Desktop\Github Projects\TravelBot---AI-Chatbot\fine_tuned_t5_travel_geography"

if REPO_ID.startswith("<"):
    raise SystemExit("Please set REPO_ID in this file before running")

print(f"Uploading {LOCAL_FOLDER} to {REPO_ID} ...")
upload_folder(
    repo_id=REPO_ID,
    repo_type="model",
    folder_path=LOCAL_FOLDER,
    path_in_repo=".",
    create_pr=False,
)
print("Upload finished. Visit https://huggingface.co/" + REPO_ID)
