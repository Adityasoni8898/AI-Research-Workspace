# 🧠 AI Research Workspace

A lightweight, AI-powered workspace for literature reviews and academic research. 

It uses Google's Gemini API for the reasoning and ChromaDB for local vector storage, keeping things fast and entirely private on your own machine.

### Main Page

<img width="2930" height="1678" alt="image" src="https://github.com/user-attachments/assets/762c3937-08fe-4d99-b13b-1263fd01ca12" />

---

## What it does

The workspace is split into three main modes:

*   **Ask (RAG Chat):** Chat with your research. Ask questions and get streamed answers backed by citations from your uploaded PDFs or imported OpenAlex papers. It falls back to general knowledge (and tells you) if the answer isn't in your library.
*   **Research:** Type in a topic, set your parameters (like "Max Papers" or "Since 2020"), and the app will automatically fetch relevant academic papers from OpenAlex and embed them into your local database. You can also drag-and-drop PDFs directly into your library.
<img width="2388" height="1486" alt="image" src="https://github.com/user-attachments/assets/fd92cb78-1126-407c-b126-ae614db8a367" />
*   **Evidence Map:** Give it a contested topic (e.g., "Impact of microplastics on marine life") and the AI will analyze your library to generate a matrix of claims, showing supporting evidence, conflicting evidence, and the current scientific consensus.
<img width="2306" height="1570" alt="image" src="https://github.com/user-attachments/assets/1d6181ae-b73c-47d6-adf9-fa9dc6a729f7" />


---

## Tech Stack

*   **Frontend:** Plain HTML, CSS, and Vanilla JavaScript (No heavy frameworks, just a single clean `index.html` file).
*   **Backend:** Python & FastAPI.
*   **AI Engine:** LangChain, Google Gemini (3.5 Flash), and gemini-embedding-2 for Embeddings.
*   **Database:** ChromaDB (Local vector storage) & simple JSON for project metadata.

---

## Getting Started

Because the database runs locally, getting this up and running is super simple.

### 1. Prerequisites
You'll need **Python 3.9+** installed on your machine and a **Google Gemini API Key** (you can get one for free from Google AI Studio).

### 2. Installation
Clone the repository and install the required Python packages:

```bash
git clone https://github.com/Adityasoni8898/AI-Research-Workspace.git
cd AI-Research-Workspace/backend

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Do make sure to have gemini api key
GEMINI_API_KEY = <your_key>