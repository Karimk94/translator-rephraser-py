import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import os

# --- Configuration ---
BASE_MODEL_PATH = "./local_gemma_model"
DATASET_PATH = "./names_dataset.jsonl"
NEW_ADAPTER_PATH = "./gemma-translator-adapter"

# --- 1. Load the dataset ---
print("Loading dataset...")
dataset = load_dataset('json', data_files=DATASET_PATH, split="train")

# --- 2. Configure Quantization (for memory efficiency) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# --- 3. Load the base model and tokenizer ---
print("Loading base model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    # --- ADDED FOR STABILITY as per warning ---
    attn_implementation="eager" 
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 4. Configure LoRA (the fine-tuning method) ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 5. Set up Training Arguments ---
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    # --- CHANGED OPTIMIZER for CPU compatibility ---
    optim="adamw_torch",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)

# --- 6. Create the Trainer ---
def formatting_prompts_func(example):
    text = f"<start_of_turn>user\nTranslate the following name to Arabic: \"{example['input']}\"<end_of_turn>\n<start_of_turn>model\n{example['output']}<end_of_turn>"
    return text

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_arguments,
    formatting_func=formatting_prompts_func
)

# --- 7. Start Fine-Tuning ---
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# --- 8. Save the newly trained adapter ---
print(f"Saving adapter to {NEW_ADAPTER_PATH}...")
trainer.model.save_pretrained(NEW_ADAPTER_PATH)
print("Adapter saved successfully.")