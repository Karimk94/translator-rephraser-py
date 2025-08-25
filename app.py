from flask import Flask, render_template, request, Response
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import os
import re
import time
from werkzeug.serving import run_simple

app = Flask(__name__)

# --- Model Loading ---
MODELS = {}
MODEL_CONFIG = {
    "translate_en_ar": {"path": "./local_model_en_ar", "local": True},
    "translate_ar_en": {"path": "./local_model_ar_en", "local": True},
    "rephrase_en":     {"path": "./local_model_rephrase_en", "local": True},
}

def load_models():
    """Loads all models into the MODELS dictionary."""
    for name, config in MODEL_CONFIG.items():
        path = config["path"]
        
        if config["local"] and not os.path.isdir(path):
            print(f"--- ERROR: Model directory not found at '{path}' ---")
            print("Please run 'download_model.py' first.")
            exit()
        
        print(f"Loading model: {name}...")
        
        if "rephrase_en" in name:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSeq2SeqLM.from_pretrained(path)
        
        MODELS[name] = {"tokenizer": tokenizer, "model": model}
    print("All models loaded successfully!")

load_models()

def is_arabic(text):
    """Detects if the text contains a significant number of Arabic characters."""
    if not text or not isinstance(text, str):
        return False
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    total_chars = len(text)
    return (arabic_chars / total_chars) > 0.1 if total_chars > 0 else False

# --- Generation Logic ---
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    original_text = data.get('text', '')
    task = data.get('task', 'translate')

    def stream_generator():
        try:
            use_live_streamer = False
            decoded_output = ""

            if task == 'translate':
                model_key = "translate_ar_en" if is_arabic(original_text) else "translate_en_ar"
                tokenizer = MODELS[model_key]["tokenizer"]
                model = MODELS[model_key]["model"]
                inputs = tokenizer(original_text, return_tensors="pt")

                outputs = model.generate(
                    **inputs, max_length=250, num_beams=5, early_stopping=True
                )
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            elif task == 'rephrase':
                if is_arabic(original_text):
                    en_tokenizer = MODELS["translate_ar_en"]["tokenizer"]
                    en_model = MODELS["translate_ar_en"]["model"]
                    ar_tokenizer = MODELS["translate_en_ar"]["tokenizer"]
                    ar_model = MODELS["translate_en_ar"]["model"]
                    
                    inputs1 = en_tokenizer(original_text, return_tensors="pt")
                    intermediate1 = en_model.generate(**inputs1, num_beams=5, do_sample=True, top_k=50)
                    english_text1 = en_tokenizer.decode(intermediate1[0], skip_special_tokens=True)
                    
                    inputs_back1 = ar_tokenizer(english_text1, return_tensors="pt")
                    final1 = ar_model.generate(**inputs_back1, max_length=250, num_beams=5)
                    decoded_output = ar_tokenizer.decode(final1[0], skip_special_tokens=True)
                else: 
                    # For English, use the Gemma model
                    use_live_streamer = True
                    model_key = "rephrase_en"
                    tokenizer = MODELS[model_key]["tokenizer"]
                    model = MODELS[model_key]["model"]
                    
                    # Create a prompt for rephrasing
                    prompt = f"""Rewrite the following text into a single, clear, and natural sentence.

**Instructions:**
1. Identify the specific names listed after "VIPs:".
2. Make these individuals the primary subject of the new sentence.
3. Replace generic nouns like 'man' or 'person' with these names.
4. Incorporate the described actions (e.g., 'sitting on a bench') and surrounding objects (e.g., 'bottle') naturally.
5. Omit all labels and unnecessary words like "VIPs:", "and also", and "person.".

**Text to rewrite:**
"{original_text}"
"""
                    inputs = tokenizer(prompt, return_tensors="pt")

                    generation_kwargs = dict(
                        **inputs, max_new_tokens=100, do_sample=True, top_k=120, top_p=0.95, temperature=1.5
                    )
            else:
                yield "data: [ERROR]\n\n"
                return

            if use_live_streamer:
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs['streamer'] = streamer
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                for new_text in streamer:
                    yield f"data: {new_text}\n\n"
            else:
                for char in decoded_output:
                    yield f"data: {char}\n\n"
                    time.sleep(0.01)
            
            yield "data: [END_OF_STREAM]\n\n"

        except Exception as e:
            print(f"Error during stream generation: {e}")
            yield f"data: [ERROR]\n\n"

    return Response(stream_generator(), mimetype='text/event-stream')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    run_simple(
        '127.0.0.1',
        5005,
        app,
        use_reloader=False,
        use_debugger=True,
        threaded=True,
        exclude_patterns=['*__pycache__*', '*venv*', '*local_model_ar_en*','*local_model_en_ar*','*local_model_rephrase_en*']
    )