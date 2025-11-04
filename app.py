from flask import Flask, render_template, request, Response
from flask_cors import CORS
import re
import requests
import json
from waitress import serve

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Ollama Configuration ---
OLLAMA_API_URL = "http://10.100.61.225:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"

def is_arabic(text):
    """Detects if the text contains a significant number of Arabic characters."""
    if not text or not isinstance(text, str):
        return False
    return bool(re.search('[\u0600-\u06FF]', text))

def remove_diacritics(text):
    """
    NEW FUNCTION: Removes all Arabic diacritics (Tashkeel) from a string.
    This provides a guaranteed fix even if the model ignores the prompt.
    """
    if not text:
        return text
    arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    return re.sub(arabic_diacritics, '', text)

def create_prompt(text, task, source_lang):
    """
    UPDATED FUNCTION: The prompt is now even stricter to prevent 
    both diacritics and '?' symbols.
    """
    if task == 'translate':
        target_lang = "Arabic" if source_lang == "English" else "English"
        
        diacritics_rule = ""
        other_rules = ""

        if target_lang == "Arabic":
            diacritics_rule = """1. **ABSOLUTELY CRITICAL**: Your response MUST NOT contain any Arabic diacritics (Tashkeel / formations). The output must be plain, simple Arabic text.
   - **Example (Bad)**: "مَرْحَبًا"
   - **Example (Good)**: "مرحبا"
2. **CRITICAL**: Do NOT output any question mark `?` characters or any other non-Arabic punctuation unless it is a direct translation of punctuation in the source text."""
            other_rules = f"""3. Your response MUST contain ONLY the translated text.
4. Do NOT add any comments, explanations, or introductory phrases.
5. **CRITICAL**: You MUST convert every single {source_lang} word and name into the {target_lang} alphabet. Your final output must not contain any {source_lang} characters."""
        else:
            other_rules = f"""1. Your response MUST contain ONLY the translated text.
2. Do NOT add any comments, explanations, or introductory phrases.
3. **CRITICAL**: You MUST convert every single {source_lang} word and name into the {target_lang} alphabet. Your final output must not contain any {source_lang} characters."""

        system_message = f"""<start_of_turn>user
You are a direct translation engine. Your task is to translate the provided {source_lang} text to {target_lang}.

Follow these rules exactly:
{diacritics_rule}
{other_rules}

### TASK ###
{source_lang} Text: "{text}"<end_of_turn>
<start_of_turn>model
"""
        return system_message
        
    elif task == 'rephrase':
        tag_lang = "Arabic" if source_lang == "Arabic" else "English"
        json_key = "arabic_tags" if source_lang == "Arabic" else "english_tags"
        
        return f"""<start_of_turn>user
You are a JSON output machine. Your only function is to output a specific JSON structure. Follow these steps exactly:

1.  Analyze this {source_lang} text: "{text}"
2.  Extract the key nouns, entities, and concepts for use as search tags.
3.  **All tags MUST be in {tag_lang}.** Do not mix languages.
4.  **Correct any spelling errors and standardize abbreviations.**
5.  Remove all stop words, non-essential words, and duplicate entries.
6.  If generating {tag_lang} tags, do NOT include any diacritics (Tashkeel / formations).
7.  Output **NOTHING** except for the completed JSON structure below. Do not use markdown. Do not add ```json. Do not add any other text.

COPY AND PASTE THIS TEMPLATE, THEN FILL IT IN:
{{"{json_key}": []}}

Your entire response must be only the filled-out template.<end_of_turn>
<start_of_turn>model
"""
    return text

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    original_text = data.get('text', '')
    task = data.get('task', 'translate')
    source_language = "Arabic" if is_arabic(original_text) else "English"
    prompt = create_prompt(original_text, task, source_language)
    
    # Data payload for the Ollama API
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True
    }

    print(f"DEBUG: Sending POST request to: {OLLAMA_API_URL}")

    def generator():
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
            response.raise_for_status()

            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = json.loads(chunk.decode('utf-8'))
                    if 'response' in decoded_chunk and decoded_chunk['response'] is not None:
                        
                        raw_response = decoded_chunk['response']
                        cleaned_response = remove_diacritics(raw_response)
                        yield f"data: {cleaned_response}\n\n"
                        
                    if decoded_chunk.get('done', False):
                        break
            
            yield "data: [END_OF_STREAM]\n\n"

        except requests.exceptions.RequestException as e:
            print(f"Error during generation: {e}")
            yield "data: [ERROR]\n\n"

    return Response(generator(), mimetype='text/event-stream')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting server on [http://0.0.0.0:5005](http://0.0.0.0:5005)")
    serve(app, host='0.0.0.0', port=5005, threads=100)