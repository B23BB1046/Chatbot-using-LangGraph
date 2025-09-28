from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import os
import sqlite3
from dotenv import load_dotenv
load_dotenv()


# ---------- PDF Reading ----------

def read_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text



def create_vectorstores(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)


class Chatstate(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pdf_path: str | None
    query: str | None
    mode: str 


# ---------- LLM Setup ----------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),  
    model="openai/gpt-oss-120b"
)




def Agent1(state: Chatstate):
    pdf_path = state.get("pdf_path", None)
    query = state.get("query", "")
    messages = state.get("messages", [])

    if not query:
        return {"messages": messages + [AIMessage(content="Please enter the question.")]}

    # If no PDF, use normal LLM
    if not pdf_path:
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="openai/gpt-oss-120b"
        )
        answer = llm.predict(
            f"You are a helpful assistant. Answer the following question:\n{query}"
        )
        return {"messages": messages + [AIMessage(content=answer)]}

    # If PDF is provided
    vectorstore = st.session_state.get("vectorstore", None)
    if vectorstore is None:
        text = read_pdf(pdf_path)
        vectorstore = create_vectorstores(text)
        st.session_state["vectorstore"] = vectorstore

    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return {"messages": messages + [AIMessage(content="Sorry, no relevant information found in PDF.")]}

    context = "\n".join([d.page_content for d in docs])

    llm1 = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )
    answer = llm1.predict(
        f"You are a helpful assistant. Answer the question based on the following context:\n{context}\n\nQuestion: {query}"
    )
    return {"messages": messages + [AIMessage(content=answer)]}

conn = sqlite3.connect(database = 'chatbot.db',check_same_thread=False)


checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(Chatstate)

graph.add_node("Agent1", Agent1)

graph.add_edge(START,'Agent1')
graph.add_edge('Agent1',END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)