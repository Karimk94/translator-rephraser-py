# AI Writing Assistant (Single-Model Version)

This is a web-based application, built with Python and Flask, that uses a locally-hosted Gemma AI model to provide advanced text manipulation capabilities. It serves as a powerful tool for translating and rephrasing text between English and Arabic using a single, efficient model.

## Features

-   **Auto-Detect Translation**: Automatically detects whether the input text is English or Arabic and translates it to the other language.
-   **Contextual Rephrasing**: Rephrases text in its original language to improve style, clarity, or tone.
-   **On-Demand Alternatives**: After an initial result is generated, you can request a different version with a single click.
-   **Live Streaming Response**: Generated text streams to the user character-by-character, creating a dynamic "typing" effect.
-   **Modern UI**: A clean, responsive user interface that is easy to use.
-   **Cancellable Actions**: Users can stop a generation task midway through the process.
-   **Fully Local**: The AI model is run directly on the local machine, ensuring privacy and offline capability after the initial setup.

## Tech Stack

-   **Backend**: Python, Flask
-   **AI Model**: Hugging Face Transformers (Google's Gemma)
-   **Frontend**: HTML5, CSS3, JavaScript (with AJAX/Fetch API)
-   **Environment**: Works with a standard Python virtual environment.

## Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Prerequisites

-   **Python 3.12**: This project requires Python version 3.12. You can download it from the official Python website. During installation, ensure you check the box that says "Add python.exe to PATH".

### 2. Clone the Repository

First, get the project files onto your machine. If you have Git, you can clone it. Otherwise, download the source code as a ZIP file and extract it.

### 3. Create a Virtual Environment

It is highly recommended to use a virtual environment to keep project dependencies isolated.

```bash
# Navigate into your project directory
cd path/to/your/project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate