
# Project Overview

This project is a Chatbot powered by LangGraph workflows and large language models. The chatbot is capable of:

💬 Answering user questions in natural language.

📂 Uploading and processing PDF documents, allowing users to ask questions based on the content of the uploaded PDF.

🧠 Using LangGraph agents to manage workflows and retrieve context-aware answers.

Currently, the chatbot integrates an OpenAI 120B parameter model. While this model does not provide real-time answers from the internet, it excels at understanding text, reasoning, and generating responses based on the provided context (such as your uploaded documents).

✨ Key Features

Question answering on both general queries and document-specific queries.

Context retrieval from PDFs for accurate and relevant answers.

Modular LangGraph-based design, making it easy to extend with new tools and agents.

Streamlit-based frontend for a simple and interactive user experience.


## API Reference

#### Get all items

```http
  https://console.groq.com/home
```

