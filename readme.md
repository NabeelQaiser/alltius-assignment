# Customer Support RAG Chatbot

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot for customer support, focused on answering queries related to insurance and AngelOne support. The bot leverages two main knowledge sources:

1. **Insurance Documents**: PDF files containing insurance policy details are embedded and indexed using FAISS and Ollama embeddings. The bot retrieves relevant information from these documents to answer insurance-related questions.
2. **AngelOne Support Webpages**: The bot crawls and extracts content from AngelOne's official support pages. It uses an LLM (Groq's Llama 3) to filter and synthesize relevant information from these webpages to answer AngelOne support queries.

The chatbot is accessible via a Streamlit web interface, providing a conversational experience for users seeking support on insurance or AngelOne account topics.

---

## Setup Instructions

### 1. Clone the Repository
```zsh
git clone <repo-url>
cd alltius-assignment
```

### 2. Create and Activate a Python Virtual Environment
```zsh
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies
```zsh
pip install -r requirements.txt
```

### 4. Set Up Groq API Key
- Sign up at [Groq](https://console.groq.com/) and obtain your API key.
- Set the environment variable in your shell:
```zsh
export GROQ_API_KEY="<your-groq-api-key>"
```

### 5. Set Up Ollama (for Embeddings)
- Install [Ollama](https://ollama.com/) and ensure it is running locally.
- Download the `llama3` model for embeddings:
```zsh
ollama pull llama3
```

### 6. Add Insurance PDFs (Optional)
- Place your insurance PDF files in the `resources/insurance-pdfs/` directory. The bot will automatically index new PDFs on first run.

---

## Running the Application

Start the Streamlit web interface:
```zsh
streamlit run chatbot_web_interface.py
```

- Access the chatbot in your browser at the URL shown in the terminal (usually [http://localhost:8501](http://localhost:8501)).

---

## Usage
- Ask questions about insurance policies, claim processes, AngelOne account support, refunds, account verification, and more.
- The bot will answer using the latest information from indexed insurance documents and AngelOne support webpages.

---

## Notes
- Ensure both the Groq API and Ollama server are accessible before running the app.
- The first run may take longer as it indexes PDFs and crawls support pages.
- For any issues, check your API keys, Ollama status, and PDF file placements.

---

## Project Structure
- `chatbot_web_interface.py`: Streamlit frontend for the chatbot.
- `chatbot_backend/agents/customer_support.py`: Main agent logic combining both retrievers.
- `chatbot_backend/tools/insurance_pdf.py`: Insurance PDF embedding and retrieval.
- `chatbot_backend/tools/angelone_support.py`: AngelOne support webpage retrieval and LLM synthesis.
- `resources/insurance-pdfs/`: Directory for insurance PDF files.
- `requirements.txt`: Python dependencies.

---

## License
This project is for demonstration and educational purposes only.
