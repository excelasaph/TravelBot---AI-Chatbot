# TravelBot — Travel & Geography Chatbot

This repository contains a domain-specific chatbot fine-tuned for travel and geography question-answering using a pre-trained T5 model. The work includes data preprocessing, model fine-tuning, evaluation, qualitative testing, and a Streamlit-based user interface for interactive demos.

## Contents
- `Travel_ChatBot_2.ipynb` — Main Jupyter Notebook with data preprocessing, tokenization, training, evaluation, and qualitative testing.
- `data/preprocessed_travel_geography_20k.csv` — Preprocessed dataset used for fine-tuning (20k English pairs).
- `fine_tuned_t5_travel_geography/` — Saved fine-tuned model and tokenizer artifacts (inference-ready).
- `metrics/evaluation_metrics.json` & `metrics/evaluation_metrics.csv` — Evaluation metrics (ROUGE, BERTScore, BLEU).
- `metrics/predictions_sample_128.csv` — Sample model predictions for qualitative review.
- `streamlit_app.py` — Streamlit web UI for interacting with the model locally.
- `requirements.txt` — Python dependencies for local setup.
- `images/` — training loss and wordclouds useful for the project report.

## Project summary
Purpose: Build a travel & geography chatbot capable of answering domain-related queries. The model was fine-tuned from `t5-base` on a curated Travel-Geography dataset (BAAI/IndustryInstruction_Travel-Geography) and evaluated with ROUGE, BERTScore, and BLEU.

Why PyTorch: Training and inference used Hugging Face's PyTorch model and Trainer (saved artifacts are PyTorch). This choice is documented and justified in the notebook due to ecosystem maturity and availability of training tools; include this justification in your submission if TensorFlow is specifically required by your instructor.

## Quick start (Local)
1. Create a virtual environment (recommended) and activate it.

PowerShell example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ensure `fine_tuned_t5_travel_geography/` is present in the repo root (contains `config.json`, `tokenizer_config.json`, and model weights).

3. Run the Streamlit app:

```powershell
streamlit run streamlit_app.py
```

The app will open in your browser. Enter travel/geography queries in the input box and press "Generate Response".

## Reproducing training and evaluation
Open `Travel_ChatBot_2.ipynb` which contains the end-to-end pipeline:
- Dataset loading and filtering (English examples sampled to 20k)
- Text normalization and lemmatization
- Tokenization using T5 tokenizer (max_length=512)
- Training using Hugging Face Trainer (training_args included)
- Evaluation with ROUGE, BERTScore, BLEU

Note: GPU is recommended for training and inference. The notebook includes GPU memory growth setup for TensorFlow but training is with PyTorch/T5. If you prefer a TensorFlow-based variant, a porting section can be added.

## Metrics (existing)
See `metrics/evaluation_metrics.json` for saved results. Key values:
- ROUGE-1: 0.3814
- ROUGE-2: 0.1824
- ROUGE-L: ~0.2958
- BERTScore (avg f1): ~0.8765
- BLEU: ~0.1088

## UI design notes
- Streamlit app is responsive and uses a two-column layout with inference settings and examples. The sidebar shows quick model metrics loaded from `metrics/evaluation_metrics.json`.
- The app includes a basic out-of-domain heuristic and a download button for the generated response.

## Deliverables for submission
- Jupyter Notebook: `Travel_ChatBot_2.ipynb`
- Model artifacts: zip the `fine_tuned_t5_travel_geography/` folder if required by submission.
- Demo video: Record 5–10 minutes showcasing the notebook, training summary, evaluation, and Streamlit UI. Upload and include link here.
- PDF report: Summarize the project, include the experiment table, metrics, and sample conversations.

## Next steps / recommendations
- Add a short hyperparameter experiment table in the notebook (lr and batch variations) to strengthen model fine-tuning rubric points.
- If submission requires TensorFlow specifically, include a short justification of PyTorch choice or port the training code.
- Host the Streamlit app (Render, Streamlit Community Cloud, or Heroku) for an accessible demo link.

## Contact
For questions about reproducing results locally or for help packaging the demo, see the notebook for the code flow and contact details in your submission PDF.