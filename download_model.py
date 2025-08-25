# download_model.py
import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# --- Dependency Check ---
try:
    import torch
    import sentencepiece
except ImportError as e:
    print(f"--- ERROR: Missing Required Library: {e.name} ---")
    print("Please install all required libraries: pip install transformers torch sentencepiece")
    sys.exit(1)

# --- Model Definitions ---
# A dictionary where the key is the local folder name and the value is the Hugging Face model name.
MODELS_TO_DOWNLOAD = {
    "local_model_en_ar": "Helsinki-NLP/opus-mt-en-ar",
    "local_model_ar_en": "Helsinki-NLP/opus-mt-ar-en",
    "local_model_rephrase_en": "google/gemma-3-1b-it",
}

def download_model(model_name, save_directory):
    """Downloads a tokenizer and model from Hugging Face and saves it locally."""
    try:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        print(f"Downloading model '{model_name}' to '{save_directory}'...")

        # Download and save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_directory)

        # Download and save the model
        if "gemma" in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        model.save_pretrained(save_directory)
        print(f"Successfully downloaded {model_name}.\n")

    except Exception as e:
        print(f"\n--- FAILED to download {model_name} ---")
        print(f"Error: {e}\n")

# --- Main Download Logic ---
if __name__ == "__main__":
    for local_dir, model_hub_name in MODELS_TO_DOWNLOAD.items():
        if os.path.exists(local_dir):
            print(f"Directory '{local_dir}' already exists. Skipping download.")
        else:
            download_model(model_hub_name, local_dir)
    
    print("All model downloads attempted.")