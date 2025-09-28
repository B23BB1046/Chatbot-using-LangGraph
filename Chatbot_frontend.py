import streamlit as st
from langgraph_backend import chatbot ,retrive_all_threads # make sure backend handles PDF/vectorstore
from langchain_core.messages import HumanMessage
import fitz
import uuid

st.set_page_config(layout="wide")
st.title("PDF Reader with Docked Chatbot")

# thread_id generator
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        
def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']

def give_name_to_thread_id(thread_id,user_input):
    topic_name = user_input[:30]+("..." if len(user_input)>30 else"")
    st.session_state['thread_names'][thread_id] = topic_name
   
# ---------------- Session State ----------------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
    
    
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] =retrive_all_threads()
add_thread(st.session_state['thread_id'])

if 'thread_names' not in st.session_state:
    st.session_state['thread_names'] = {}

    
# -- sidebar
st.sidebar.title('Yogendra chatbot')
if st.sidebar.button('new bakchodi'):
    reset_chat()

st.sidebar.header('Meri bakchodiyan')
for thread_id in st.session_state['chat_threads'][::-1]:
    thread_label = st.session_state['thread_names'].get(thread_id,str(thread_id))
    if st.sidebar.button(f"Thread{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role':role,'content':msg.content})
        st.session_state['message_history'] = temp_messages

if "pdf_path" not in st.session_state:
    st.session_state["pdf_path"] = None

# ---------------- Sidebar PDF Selector ----------------
uploaded_file = st.sidebar.file_uploader("Select a PDF to read", type=["pdf"])
if uploaded_file:
    pdf_path = f"temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state["pdf_path"] = pdf_path
    st.sidebar.success("PDF processed!")

# ---------------- Layout: PDF Viewer + Chat ----------------
pdf_col, chat_col = st.columns([2, 1])  # PDF wide, chat docked right

# ---------------- PDF Viewer ----------------
with pdf_col:
    if st.session_state["pdf_path"]:
        doc = fitz.open(st.session_state["pdf_path"])
        for i, page in enumerate(doc):
            st.markdown(f"**Page {i+1}**")
            st.write(page.get_text())
    else:
        st.info("Upload a PDF from the sidebar to read.")

# ---------------- Docked Chatbot ----------------
with chat_col:
    st.subheader("Chatbot ")
    for msg in st.session_state["message_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question...")
    if user_input:
        if st.session_state['thread_id'] not in st.session_state['thread_names']:
            give_name_to_thread_id(st.session_state['thread_id'],user_input)
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Only pass pdf_path & query; do NOT pass FAISS/vectorstore
        state = {
            "messages": [HumanMessage(content=user_input)],
            "pdf_path": st.session_state.get("pdf_path"),
            "query": user_input,
            "mode": "PDF Mode" if st.session_state.get("pdf_path") else "Chat Mode",
        }

        # Backend should process PDF & create/retrieve vectorstore internally
        #response = chatbot.invoke(state)
        #ai_message = response["messages"][-1].content

        #st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
        with st.chat_message("assistant"):
            full_response = []
            #ai_message = st.write_stream(
            def stream_response():
                
                for message_chunk, metadata in chatbot.stream(
                    state,
                    config={'configurable':{'thread_id':st.session_state['thread_id']}},
                    stream_mode='messages'
                ):
                    if hasattr(message_chunk,"content"):
                        text = message_chunk.content  
                        full_response.append(text)
                        yield text
                
                
                #message_chunk.content for message_chunk, metadata in chatbot.stream(
                    
                    #state,  # <-- use the full state with pdf_path, query, etc.
                    
                    
                    #config={'configurable': {'thread_id': st.session_state['thread_id']}},
                    
                    #stream_mode='messages'
                    
                    #)
                
                #)

            #st.markdown(ai_message)
            #st.session_state['message_history'].append({'role':'assistant','content':ai_message})
            st.write_stream(stream_response())
            final_text = "".join(full_response)
            st.session_state['message_history'].append({"role":"assistant","content":final_text})
