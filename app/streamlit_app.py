import streamlit as st
import os
import json
import urllib.parse
import random
from datetime import datetime
try:
    # Preferred import when available
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _TRANSFORMERS_AUTO = True
except Exception:
    # Older/newer transformers builds may not expose Auto* names in the same place.
    # Fall back to T5-specific classes which are appropriate for a fine-tuned T5 model.
    from transformers import T5Tokenizer as AutoTokenizer, T5ForConditionalGeneration as AutoModelForSeq2SeqLM
    _TRANSFORMERS_AUTO = False

st.set_page_config(page_title="TravelBot Demo", page_icon=":airplane:", layout="wide", initial_sidebar_state="expanded")

st.title("TravelBot: Travel & Cultural Routes — Demo")

# Remote model server (recommended for Spaces). If provided in Secrets as MODEL_SERVER_URL,
# the app will call the external server and avoid loading large model files inside the Space.
MODEL_SERVER_URL = None
try:
    MODEL_SERVER_URL = st.secrets.get("MODEL_SERVER_URL")
except Exception:
    MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL")

MODEL_ID = st.sidebar.text_input("Hugging Face model repo (user/model)", value="excelasaph/fine_tuned_t5_travel_geography")
use_recommended = st.sidebar.checkbox("Use recommended generation settings", True)
# Add a beams control (safe, non-disruptive)
num_beams = st.sidebar.slider("Beams (creativity vs accuracy)", 1, 8, 4)
# Map max tokens to model max_length
max_length = st.sidebar.slider("Max length", 50, 512, 180)
# OOD (out-of-domain) length threshold for a simple heuristic
ood_threshold = st.sidebar.slider("OOD length threshold (words)", 5, 60, 20)

@st.cache_resource
def load_model(model_id, local=False):
    # Use the tokenizer/model classes imported above. If transformers exposes Auto* names
    # this will behave as before; otherwise we use the T5-specific classes aliased above.
    
    # Add legacy=True to address the add_prefix_space warning for T5 tokenizer
    # and add silent exceptions for any tokenizer warnings to prevent startup issues
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local, legacy=True)
    except Exception as e:
        st.warning(f"Tokenizer loaded with warnings (non-blocking): {e}")
        # Fall back without legacy option if not supported
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, local_files_only=local)
    return tokenizer, model


def generate_via_server(prompt, max_length=256, num_beams=1):
    """Call remote model server (POST /generate) and return generated_text or raise."""
    import requests
    if not MODEL_SERVER_URL:
        raise RuntimeError("MODEL_SERVER_URL is not configured")
    payload = {"inputs": prompt, "max_length": max_length, "num_beams": num_beams}
    headers = {}
    # Optional: support an API key sent via secrets
    try:
        api_key = st.secrets.get("MODEL_SERVER_API_KEY")
    except Exception:
        api_key = os.environ.get("MODEL_SERVER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.post(MODEL_SERVER_URL.rstrip('/') + "/generate", json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("generated_text", "")


### Chat history helpers (centralized, safe)
CHAT_FILE = os.path.join(os.path.dirname(__file__), 'chat_history.json')

def load_chat_history():
    try:
        if os.path.exists(CHAT_FILE):
            with open(CHAT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception:
        pass
    return []


def save_chat_history(hist):
    try:
        with open(CHAT_FILE, 'w', encoding='utf-8') as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def delete_chat_history():
    try:
        if os.path.exists(CHAT_FILE):
            os.remove(CHAT_FILE)
    except Exception:
        pass


def safe_rerun():
    """Attempt to rerun the Streamlit script; fall back safely if unavailable."""
    try:
        # preferred, but may not exist in some Streamlit builds
        return st.experimental_rerun()
    except Exception:
        # Toggle a dummy session_state key so the page re-evaluates on next event,
        # then stop this run. This avoids raising AttributeError on older/new
        # Streamlit versions that don't expose experimental_rerun.
        st.session_state['_need_rerun'] = not st.session_state.get('_need_rerun', False)
        return st.stop()

if MODEL_ID and MODEL_ID != "<YOUR_USERNAME>/<YOUR_MODEL_REPO>":
    try:
        # Deploying with Streamlit: prefer local-only model loading. Remove
        # Hub-download toggle to avoid unexpected network downloads at runtime.
        tokenizer, model = load_model(MODEL_ID, local=True)
        st.success(f"Loaded model {MODEL_ID}")
    except Exception as e:
        st.error(f"Could not load model {MODEL_ID}: {e}")
        st.stop()
else:
    st.info("Enter your model repo id in the sidebar to load the model from the Hub.")
    st.stop()

col1, col2 = st.columns([2,1])

# Shared generator helper used by the Generate button and example buttons
def generate_and_display(prompt_text, out_container):
    """Generate into the provided output container to keep the widget tree stable.

    Writing results into `out_container` prevents appending widgets at the
    call site, which stabilizes layout across reruns and avoids duplication.
    """
    st.session_state['last_prompt'] = prompt_text
    out_container.empty()
    # Write all generation UI into the reserved placeholder directly. Using
    # the existing container as the context (instead of creating a nested
    # container) avoids transient layout reflows that can produce the
    # faint duplicated widgets observed on first-generation.
    with out_container:
        with st.spinner("Generating..."):
            try:
                # Append the user's message to persistent chat history
                try:
                    chat_file = os.path.join(os.path.dirname(__file__), 'chat_history.json')
                    if os.path.exists(chat_file):
                        with open(chat_file, 'r', encoding='utf-8') as f:
                            hist = json.load(f)
                    else:
                        hist = []
                except Exception:
                    hist = []
                user_entry = {'role': 'user', 'content': prompt_text, 'time': datetime.now().strftime('%I:%M %p')}
                hist.append(user_entry)
                try:
                    with open(chat_file, 'w', encoding='utf-8') as f:
                        json.dump(hist, f, ensure_ascii=False, indent=2)
                except Exception:
                    # ignore persistence failures
                    pass
                gen_kwargs = {
                    'num_beams': num_beams if not use_recommended else 4,
                    'repetition_penalty': 1.8 if use_recommended else 1.0,
                    'length_penalty': 1.0 if use_recommended else 1.0,
                    'no_repeat_ngram_size': 3 if use_recommended else 0,
                    'early_stopping': True,
                    'max_length': max_length
                }
                formatted_prompt = prompt_text if prompt_text.lower().startswith('question:') else f"question: {prompt_text} answer:"
                inputs = tokenizer(formatted_prompt, return_tensors='pt', truncation=True, padding=True)
                out = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], **gen_kwargs)
                text = tokenizer.decode(out[0], skip_special_tokens=True)

                def is_out_of_domain(generated_response, threshold_length=20):
                    return len(generated_response.split()) < threshold_length

                out_container.markdown("### Response")
                if is_out_of_domain(text, threshold_length=ood_threshold):
                    fallback = "I can only answer questions about travel and geography. Please ask a question related to that topic."
                    out_container.warning("Model response flagged as potentially out-of-domain; returning safe fallback.")
                    out_container.write(fallback)
                    final_text = fallback
                else:
                    out_container.write(text)
                    final_text = text
                # Append assistant response to persistent chat history
                try:
                    try:
                        if 'hist' not in locals():
                            chat_file = os.path.join(os.path.dirname(__file__), 'chat_history.json')
                            if os.path.exists(chat_file):
                                with open(chat_file, 'r', encoding='utf-8') as f:
                                    hist = json.load(f)
                            else:
                                hist = []
                    except Exception:
                        hist = []
                    bot_entry = {'role': 'bot', 'content': final_text, 'time': datetime.now().strftime('%I:%M %p')}
                    hist.append(bot_entry)
                    with open(chat_file, 'w', encoding='utf-8') as f:
                        json.dump(hist, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                # Render a high-quality inline SVG download icon as an anchor that
                # triggers a file download via a data URI. Place it directly under
                # the response and *before* the separator so the bar remains the
                # bottom boundary of the response block.
                try:
                    encoded = urllib.parse.quote(final_text)
                    svg_icon = (
                        "<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'>"
                        "<path d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/>")
                    svg_icon += ("<polyline points='7 10 12 15 17 10'/><line x1='12' y1='15' x2='12' y2='3'/></svg>")
                    html = f"<a download='travelbot_response.txt' href='data:text/plain;charset=utf-8,{encoded}' title='Download response (txt)' style='text-decoration:none;color:inherit'>{svg_icon}</a>"
                    out_container.markdown(html, unsafe_allow_html=True)
                except Exception:
                    # Fallback to a simple download_button if encoding or HTML injection fails
                    out_container.download_button("Download", data=final_text, file_name="travelbot_response.txt", help="Download response (txt)")
                out_container.markdown("---")
            except Exception as e:
                out_container.error(f"Generation failed: {e}")

# Left column: sample prompts (placed before the input box so they can set the session value)

# Defensive session state defaults to avoid first-run layout variation
if 'last_prompt' not in st.session_state:
    st.session_state['last_prompt'] = ''
if 'do_generate_now' not in st.session_state:
    st.session_state['do_generate_now'] = False

with col1:
    # Reserve a stable container for response output. Always create it before
    # any conditional widgets so the widget ordering doesn't change between
    # reruns (this prevents the translucent duplication on first generation).
    response_placeholder = st.container()
    # Large pool of prompts
    prompt_pool = [
        "What are the must-see attractions in Rome for a 2-day trip?",
        "How can I prepare for high-altitude hiking in the Himalayas?",
        "Suggest a 5-day itinerary for southern Italy focusing on food and history.",
        "What vaccinations are recommended before traveling to Southeast Asia?",
        "What are the best travel tips for solo female travelers in South America?",
        "How do I get from Paris to Barcelona by train?",
        "What are the top UNESCO World Heritage sites in Africa?",
        "Recommend a scenic road trip route in New Zealand.",
        "What are the visa requirements for visiting Japan as a US citizen?",
        "Suggest a family-friendly itinerary for a week in Australia.",
        "What are the best months to visit Iceland for the Northern Lights?",
        "How can I travel sustainably in Southeast Asia?",
        "What are the must-try foods in Morocco?",
        "How do I avoid altitude sickness in Peru?",
        "What are the safest countries for LGBTQ+ travelers?",
        "What are the best hiking trails in Patagonia?",
        "How do I plan a budget backpacking trip across Europe?",
        "What are the top wildlife experiences in Africa?",
        "How do I get around Italy without a car?",
        "What are the best islands to visit in Greece?",
        "How do I find authentic local experiences when traveling?",
    ]
    # Show 4 random prompts each time
    examples = random.sample(prompt_pool, 4)
    with st.expander("Sample prompts"):
        cols_ex = st.columns(2)
        for i, ex in enumerate(examples):
            key = f"sample_ex_{i}"
            if cols_ex[i % 2].button(ex, key=key):
                st.session_state['prompt_input'] = ex
                st.session_state['do_generate_now'] = True

    # Session-keyed text area uses the pre-set prompt_input when present
    if 'prompt_input' not in st.session_state:
        st.session_state['prompt_input'] = (
            "Imagine you are a tourist planning a trip to Europe and you're interested in exploring the cultural routes mentioned in the text. "
            "Provide a suggested itinerary for a 2-week trip that incorporates at least three of the cultural routes mentioned, including accommodations, transportation, and activities"
        )
    prompt = st.text_area("Enter a question / prompt", height=220, key='prompt_input')

    # Always-visible Generate button (separate from the helper function)
    if st.button("Generate"):
        # Use the current prompt value from the widget; do NOT assign back to
        # session_state['prompt_input'] (Streamlit forbids modifying a key
        # after the widget with that key has been instantiated).
        generate_and_display(prompt, response_placeholder)

    # If an example set the auto-generate flag earlier in this run, call generator now
    if st.session_state.pop('do_generate_now', False):
        generate_and_display(st.session_state.get('prompt_input', ''), response_placeholder)

with col2:
    st.markdown("## Diagnostics")
    st.write("Use recommended settings:", use_recommended)
    st.write("Max length:", max_length)
    st.write("Model repo:", MODEL_ID)

    # Chat History viewer (read-only, non-invasive)
    with st.expander("Chat History"):
        hist = load_chat_history()
        if not hist:
            st.info("No chat history saved yet.")
        else:
            # Render a compact table-like view
            for i, entry in enumerate(hist[::-1]):
                # recent-first
                role = entry.get('role', 'user')
                content = entry.get('content', '')
                t = entry.get('time', '')
                st.markdown(f"**{role.title()}** <span style='color:var(--muted)'>· {t}</span>", unsafe_allow_html=True)
                st.write(content)
                if i < len(hist) - 1:
                    st.markdown("---")

        col_del1, col_del2 = st.columns([3,1])
        with col_del1:
            if st.button('Delete chat history', key='delete_chat_history_btn'):
                # perform deletion immediately
                delete_chat_history()
                st.session_state['chat_deleted'] = True
                st.success('Chat history deleted')
        with col_del2:
            st.empty()
