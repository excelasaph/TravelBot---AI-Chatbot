#!/usr/bin/env python
"""
Troubleshooting script to verify model loading works outside of Streamlit.
Run this script directly to test tokenizer and model loading without Streamlit overhead.
"""
import os
import sys
import traceback

def test_model_loading(model_path=None, local=True):
    """Test loading the tokenizer and model and report any issues."""
    print("\n=== Testing Model Loading ===")
    model_id = model_path or "excelasaph/fine_tuned_t5_travel_geography"
    print(f"Testing model loading: {model_id} (local_files_only={local})")
    
    # Check for transformers
    try:
        import transformers
        print(f"✅ transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")
        return False
    
    # Check for torch
    try:
        import torch
        print(f"✅ torch version: {torch.__version__}")
    except ImportError:
        print("❌ torch not installed. Install with: pip install torch")
        return False
    
    # Try importing AutoTokenizer and AutoModelForSeq2SeqLM
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print("✅ Successfully imported AutoTokenizer and AutoModelForSeq2SeqLM")
    except ImportError:
        print("❌ Failed to import AutoTokenizer or AutoModelForSeq2SeqLM")
        # Try T5-specific classes
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            print("✅ Successfully imported T5Tokenizer and T5ForConditionalGeneration")
        except ImportError:
            print("❌ Failed to import T5Tokenizer or T5ForConditionalGeneration")
            return False
    
    # Try loading tokenizer first
    try:
        with open(os.devnull, 'w') as f:
            # Save original stderr to restore later
            original_stderr = sys.stderr
            # Temporarily redirect stderr to devnull
            sys.stderr = f
            
            # Try first with legacy flag
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local, legacy=True)
                print("✅ Tokenizer loaded (with legacy=True)")
            except Exception:
                # Fall back without legacy flag
                tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local)
                print("✅ Tokenizer loaded (without legacy flag)")
            
            # Restore stderr
            sys.stderr = original_stderr
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        traceback.print_exc()
        return False
    
    # Try loading model
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, local_files_only=local)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        traceback.print_exc()
        return False
    
    # Try a basic inference test
    try:
        test_input = "question: What are the best places to visit in Paris? answer:"
        inputs = tokenizer(test_input, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"], max_length=50)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Inference test passed. Output: {result[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check for local model first, fallback to HF Hub
    model_path = None
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    local_model_folder = os.path.join(repo_root, "fine_tuned_t5_travel_geography")
    
    print("\n=== Model Path Checks ===")
    print(f"Looking for local model at: {local_model_folder}")
    
    if os.path.exists(local_model_folder) and os.path.isdir(local_model_folder):
        model_config = os.path.join(local_model_folder, "config.json")
        if os.path.exists(model_config):
            print(f"✅ Found local model at: {local_model_folder}")
            model_path = local_model_folder
        else:
            print(f"⚠️ Found folder but missing config.json: {local_model_folder}")
    else:
        print(f"⚠️ Local model folder not found: {local_model_folder}")
    
    # Test loading
    success = test_model_loading(model_path, local=bool(model_path))
    
    if success:
        print("\n✅ All tests passed! Model should work in Streamlit app.")
        print("\nIf you still have issues in Streamlit, try:")
        print("1. Restarting your terminal/shell")
        print("2. Checking if any other Streamlit apps are running (port conflict)")
        print("3. Verify your virtual environment is active with the right packages")
    else:
        print("\n❌ Some tests failed. Fix the issues above before running Streamlit.")