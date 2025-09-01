from flask import Flask, render_template, request, Response
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import os
import re
from werkzeug.serving import run_simple
import time

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
        MODEL_PATH,
        device_map="auto"
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

### EXAMPLE ###
{source_lang} Text: "a man sitting on a bench. VIPs : Sheikh Mohammed Bin Zayed Al Nahyan, Sheikh Mohammed Bin Rashid Al Maktoum, Mattar Al Tayer"
{target_lang} Text:  "رجل يجلس على مقعد. الشخصيات المهمة: الشيخ محمد بن زايد، الشيخ محمد بن راشد آل مكتوم، مطر الطاير"

### TASK ###
{source_lang} Text: "{text}"<end_of_turn>
<start_of_turn>model
"""
        return system_message
        
    elif task == 'rephrase':
        return f"""<start_of_turn>user
You are a JSON output machine. Your only function is to output a specific JSON structure. Follow these steps exactly:

1.  Analyze this text: "{text}"
2.  Extract key nouns, entities, and concepts for use as search tags.
3.  Remove all stop words, non-essential words, and duplicate entries.
4.  Translate the final list of English tags into Modern Standard Arabic.
5.  Output **NOTHING** except for the completed JSON structure below. Do not use markdown. Do not add ```json. Do not add any other text.

COPY AND PASTE THIS TEMPLATE, THEN FILL IT IN:
{{"english_tags": [], "arabic_tags": []}}

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
    
    # Tokenize the entire prompt
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

    if task == 'translate':
        def translation_generator():
            try:
                # Generate only the response (after the prompt)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Decode only the NEW tokens (not the prompt)
                new_tokens = outputs[0][inputs.input_ids.shape[1]:]
                translated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                # Clean up any special tokens
                translated_text = translated_text.replace("<end_of_turn>", "").strip()
                
                # Simulate streaming
                for char in translated_text:
                    yield f"data: {char}\n\n"
                    time.sleep(0.01)
                
                yield "data: [END_OF_STREAM]\n\n"
                
            except Exception as e:
                print(f"Error during translation generation: {e}")
                yield "data: [ERROR]\n\n"
        return Response(translation_generator(), mimetype='text/event-stream')

    else: # task == 'rephrase'
        def rephrase_streamer():
            try:
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=250,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.8
                )
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                for new_text in streamer:
                    yield f"data: {new_text}\n\n"
                yield "data: [END_OF_STREAM]\n\n"
            except Exception as e:
                print(f"Error during rephrase generation: {e}")
                yield "data: [ERROR]\n\n"
        return Response(rephrase_streamer(), mimetype='text/event-stream')
    
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
        exclude_patterns=['*__pycache__*', '*venv*', '*local_gemma_model*']
    )