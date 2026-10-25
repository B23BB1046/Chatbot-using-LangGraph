# Document-Aware Chatbot with LangGraph

A conversational AI prototype that combines an LLM with a LangGraph workflow and PDF question answering. It supports general questions and questions grounded in user-provided documents, with a Streamlit-based interface.

## Features

- 💬 Natural-language chat interface.
- 📄 PDF upload and document-grounded question answering.
- 🧭 LangGraph-based workflow orchestration.
- 🧩 Separate backend and frontend modules for easier iteration.
- 🔌 LLM integration through a provider API.

## Architecture at a glance

```text
User
 └── Streamlit frontend (Chatbot_frontend.py)
       └── Chatbot workflow / backend (Chatbot_backend.py)
             ├── LLM provider
             └── PDF context and response generation
```

## Repository structure

| File | Purpose |
|---|---|
| `Chatbot_frontend.py` | User-facing chat interface |
| `Chatbot_backend.py` | Chat workflow and model integration |
| `README.md` | Project documentation |

## Getting started

1. Clone the repository:
   ```bash
   git clone https://github.com/B23BB1046/Chatbot-using-LangGraph.git
   cd Chatbot-using-LangGraph
   ```
2. Create and activate a virtual environment.
3. Install the packages imported by the Python files. Add a `requirements.txt` for a reproducible setup.
4. Set the required model-provider API key as an environment variable; never commit API keys.
5. Start the frontend:
   ```bash
   streamlit run Chatbot_frontend.py
   ```

## Tech stack

Python · LangGraph · LLM API · Streamlit · PDF question answering

## Limitations and next steps

- Answers depend on the quality of document extraction and retrieved context.
- Validate citations and factual claims before relying on generated answers.
- Add dependency pinning, structured logging, error handling, and example screenshots.
- Evaluate answer quality with a small set of test questions and expected evidence.

## Security

Store credentials in environment variables or a local `.env` file excluded by `.gitignore`. Do not upload private PDFs or secrets to a public repository.
