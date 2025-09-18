from flask import Flask, render_template, request, Response
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import os
import re
from waitress import serve

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

# --- Model Loading ---
MODEL = {}
MODEL_PATH = "./local_gemma_finetuned" 

def load_model():
    """Loads the fine-tuned Gemma model and tokenizer."""
    if not os.path.isdir(MODEL_PATH):
        print(f"--- ERROR: Model directory not found at '{MODEL_PATH}' ---")
        print("Please run 'fine_tune.py' and 'merge_model.py' first.")
        exit()
    
    print("Loading fine-tuned Gemma model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH
    )
    MODEL["tokenizer"] = tokenizer
    MODEL["model"] = model
    print("Fine-tuned model loaded successfully!")

load_model()

def is_arabic(text):
    """Detects if the text contains a significant number of Arabic characters."""
    if not text or not isinstance(text, str):
        return False
    return bool(re.search('[\u0600-\u06FF]', text))

def create_prompt(text, task, source_lang):
    """Creates a specific prompt for the given task and language, formatted for Gemma-3-1B-IT."""
    if task == 'translate':
        target_lang = "Arabic" if source_lang == "English" else "English"
    
        system_message = f"""<start_of_turn>user
You are a direct translation engine. Your task is to translate the provided {source_lang} text to {target_lang}.

Follow these rules exactly:
1. Your response MUST contain ONLY the translated text.
2. Do NOT add any comments, explanations, or introductory phrases.
3. **CRITICAL**: You MUST convert every single {source_lang} word and name into the {target_lang} alphabet. Your final output must not contain any {source_lang} characters.

### TASK ###
{source_lang} Text: "{text}"<end_of_turn>
<start_of_turn>model
"""
        return system_message
        
    elif task == 'rephrase':
        return f"""<start_of_turn>user
You are a JSON output machine. Your only function is to output a specific JSON structure. Follow these steps exactly:

1.  Analyze this text: "{text}"
2.  Extract the key nouns, entities, and concepts for use as search tags.
3.  **Correct any spelling errors and standardize abbreviations.**
4.  Remove all stop words, non-essential words, and duplicate entries.
5.  Output **NOTHING** except for the completed JSON structure below. Do not use markdown. Do not add ```json. Do not add any other text.

COPY AND PASTE THIS TEMPLATE, THEN FILL IT IN:
{{"english_tags": []}}

Your entire response must be only the filled-out template.<end_of_turn>
<start_of_turn>model
"""
    return text
# --- Generation Logic ---

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    original_text = data.get('text', '')
    task = data.get('task', 'translate')
    tokenizer = MODEL["tokenizer"]
    model = MODEL["model"]
    source_language = "Arabic" if is_arabic(original_text) else "English"
    prompt = create_prompt(original_text, task, source_language)
    
    inputs = tokenizer(prompt, return_tensors="pt") 
    inputs = inputs.to(model.device)

    def generator():
        try:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            gen_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "streamer": streamer,
                "max_new_tokens": 250,
                "do_sample": True,
                "top_k": 50,
                "pad_token_id": tokenizer.eos_token_id
            }
            if task == 'translate':
                gen_kwargs['do_sample'] = False
                gen_kwargs['max_new_tokens'] = 100


            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            for new_text in streamer:
                yield f"data: {new_text}\n\n"
            yield "data: [END_OF_STREAM]\n\n"

        except Exception as e:
            print(f"Error during generation: {e}")
            yield "data: [ERROR]\n\n"

    return Response(generator(), mimetype='text/event-stream')
    
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    serve(app, host='127.0.0.1', port=5005, threads=100)