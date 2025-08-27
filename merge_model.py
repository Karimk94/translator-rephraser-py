import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- Configuration ---
BASE_MODEL_PATH = "./local_gemma_model"
ADAPTER_PATH = "./gemma-translator-adapter"
MERGED_MODEL_PATH = "./local_gemma_finetuned"

# --- Load the base model ---
print(f"Loading base model from {BASE_MODEL_PATH}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# --- Load the LoRA adapter ---
print(f"Loading adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# --- Merge the adapter into the model ---
print("Merging model and adapter...")
model = model.merge_and_unload()

# --- Save the merged model and tokenizer ---
print(f"Saving merged model to {MERGED_MODEL_PATH}...")
model.save_pretrained(MERGED_MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print("Merged model saved successfully!")