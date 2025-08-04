AI Writing Assistant
This is a web-based application, built with Python and Flask, that uses locally-hosted AI models to provide advanced text manipulation capabilities. It serves as a powerful tool for translating and rephrasing text between English and Arabic.
Features
Auto-Detect Translation: Automatically detects whether the input text is English or Arabic and translates it to the other language.
Contextual Rephrasing: Rephrases text in its original language to improve style, clarity, or tone.
On-Demand Alternatives: After an initial result is generated, you can request a different version with a single click.
Live Streaming Response: Generated text streams to the user character-by-character, creating a dynamic "typing" effect.
Modern UI: A clean, responsive user interface that is easy to use.
Cancellable Actions: Users can stop a generation task midway through the process.
Fully Local: All AI models are run directly on the local machine, ensuring privacy and offline capability after the initial setup.
Tech Stack
Backend: Python, Flask
AI Models: Hugging Face Transformers (Helsinki-NLP for translation, Pegasus for rephrasing)
Frontend: HTML5, CSS3, JavaScript (with AJAX/Fetch API)
Environment: Works with a standard Python virtual environment.
Setup and Installation
Follow these steps to get the application running on your local machine.
1. Prerequisites
Python 3.12: This project requires Python version 3.12. You can download it from the official Python website. During installation, ensure you check the box that says "Add python.exe to PATH".
2. Clone the Repository
First, get the project files onto your machine. If you have Git, you can clone it. Otherwise, download the source code as a ZIP file and extract it.
3. Create a Virtual Environment
It is highly recommended to use a virtual environment to keep project dependencies isolated.
# Navigate into your project directory
cd path/to/your/project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


4. Install Dependencies
Create a file named requirements.txt in the root of your project directory and paste the following content into it:
flask
transformers
torch
sentencepiece
protobuf==3.20.3
sacremoses


Now, install all the required packages with a single command:
pip install -r requirements.txt


5. Download the AI Models
The application uses several AI models that need to be downloaded. Run the provided Python script to do this automatically. This only needs to be done once.
python download_model.py


This will create several local_model_* folders in your project directory.
6. Run the Application
You are now ready to start the web server.
flask run


The terminal will show that the models are loading and will eventually display a message like:
* Running on http://127.0.0.1:5000
How to Use
Open your web browser and navigate to http://127.0.0.1:5000.
Enter English or Arabic text into the text area.
Click "Translate" to translate the text to the other language.
Click "Rephrase" to get a different wording of the text in the same language.
After a result appears, click the "regenerate" icon (🔄) to get a new alternative.
Click "Stop Generating" at any time to cancel the current task.
Project Structure
.
├── static/
│   ├── style.css       # All CSS styles
│   └── script.js       # All frontend JavaScript logic
├── templates/
│   └── index.html      # The main HTML user interface
├── local_model_en_ar/  # Folder for the downloaded En->Ar model
├── local_model_ar_en/  # Folder for the downloaded Ar->En model
├── app.py              # The main Flask application and backend logic
├── download_model.py   # Script to download the required AI models
└── requirements.txt    # List of Python dependencies


