"""
FastAPI backend for TravelBot model.
Exposes a POST /generate endpoint that accepts JSON payload:
{
  "inputs": "your prompt string",
  "max_length": 180,
  "num_beams": 4
}

The API will load the local `fine_tuned_t5_travel_geography/` model if present and perform generation.

Run locally with:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

The Streamlit app can call POST http://<host>:8000/generate with the JSON body above.
"""
from typing import Optional
import os
import logging
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception as e:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="TravelBot API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8501", "http://127.0.0.1", "http://127.0.0.1:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
LOCAL_MODEL_FOLDER = os.path.join(REPO_ROOT, "fine_tuned_t5_travel_geography")

_model = None
_tokenizer = None
_model_source = None


class GenerateRequest(BaseModel):
    inputs: str = Field(..., description="Prompt text to generate from")
    max_length: Optional[int] = Field(180, ge=10, le=1024, description="Maximum generation length")
    num_beams: Optional[int] = Field(4, ge=1, le=16, description="Beam search width")


class GenerateResponse(BaseModel):
    generated_text: str
    model_source: Optional[str] = None


def _ensure_transformers_available():
    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        raise RuntimeError("transformers is not available. Install 'transformers' in your environment.")


def load_model(local_folder=LOCAL_MODEL_FOLDER):
    """Load tokenizer and model. Returns (tokenizer, model, source, path)
    source is one of 'local-folder', 'hub-cache', or 'hub'."""
    global _model, _tokenizer, _model_source

    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model, _model_source, local_folder if os.path.isdir(local_folder) else None

    _ensure_transformers_available()

    try:
        if os.path.isdir(local_folder):
            logger.info("Loading model from local folder: %s", local_folder)
            try:
                tokenizer = AutoTokenizer.from_pretrained(local_folder, local_files_only=True, legacy=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(local_folder, local_files_only=True)
                
            model = AutoModelForSeq2SeqLM.from_pretrained(local_folder, local_files_only=True)
            _tokenizer, _model, _model_source = tokenizer, model, 'local-folder'
            return tokenizer, model, 'local-folder', local_folder
    except Exception as e:
        logger.warning("Local folder load failed: %s", e)

    try:
        # Load using the folder name as repo id 
        repo_id = os.path.basename(local_folder)
        logger.info("Attempting to load model via repo id: %s", repo_id)
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id, local_files_only=True, legacy=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
            
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, local_files_only=True)
        _tokenizer, _model, _model_source = tokenizer, model, 'hub-cache'
        return tokenizer, model, 'hub-cache', None
    except Exception as e:
        logger.warning("Hub-cache load failed: %s", e)

    try:
        repo_id = os.path.basename(local_folder)
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id, legacy=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
        _tokenizer, _model, _model_source = tokenizer, model, 'hub'
        return tokenizer, model, 'hub', None
    except Exception as e:
        logger.exception("Could not load model from local folder or hub: %s", e)
        raise RuntimeError("Model not available locally or on Hugging Face Hub. Place the model in 'fine_tuned_t5_travel_geography/' or ensure network access.")


@app.on_event("startup")
async def startup_event():
    port = os.environ.get("PORT", "8000")
    logger.info(f"Starting app on port {port}")

    try:
        load_model()
        logger.info("Model loaded at startup: %s", _model_source)
    except Exception as e:
        logger.warning("Model not loaded at startup (deferred until first request): %s", e)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generates text from a given prompt. Returns generated_text and model_source.

    Robust error handling: validates request, loads model lazily if needed, and returns clear HTTP errors.
    """
    if not req.inputs or not isinstance(req.inputs, str):
        raise HTTPException(status_code=400, detail="'inputs' must be a non-empty string")

    try:
        tokenizer, model, model_source, model_path = load_model()
    except Exception as e:
        logger.error("Model load failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    try:
        prompt = req.inputs if req.inputs.lower().startswith('question:') else f"question: {req.inputs} answer:"
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
        gen_kwargs = {
            'num_beams': req.num_beams,
            'repetition_penalty': 1.8,
            'length_penalty': 1.0,
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
            'max_length': req.max_length,
        }
        out = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], **gen_kwargs)
        text = tokenizer.decode(out[0], skip_special_tokens=True)

        # Simple out-of-domain heuristic (same as Streamlit)
        if len(text.split()) < 5:
            final_text = "I can only answer questions about travel and geography. Please ask a question related to that topic."
        else:
            final_text = text

        return GenerateResponse(generated_text=final_text, model_source=model_source)
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@app.get("/")
async def root():
    return {"status": "ok", "model_loaded": _model is not None, "model_source": _model_source}
