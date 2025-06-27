import os
import time
import tempfile

import streamlit as st 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory

# Langchain core classes and utilities 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Langchain LLM and chaining utilities
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# Text splitting & embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector Store
from langchain_community.vectorstores import Chroma

# PDF file loader
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="🩺 AI Medical Chatbot", layout="wide", initial_sidebar_state="expanded")
st.title("🩺 AI Medical Chatbot - PDF Q&A with Chat History")

st.sidebar.header("Configuration")
st.sidebar.write(
    "- Enter your GROQ API Key\n"
    "- Upload Medical PDFs only (e.g., clinical guides, research papers)\n"
    "- Ask medical questions only"
)

# API and embedding setup
api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not api_key:
    st.warning("🔑 Please enter your Groq API Key to continue.")
    st.stop()

# Load LLM
llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")

# Upload PDFs
uploaded_files = st.file_uploader(
    "📥 Upload medical-related PDF file(s)",
    type="pdf",
    accept_multiple_files=True,
)

all_docs = []

if uploaded_files:
    with st.spinner("📚 Processing and splitting PDFs..."):
        for pdf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(pdf.getvalue())
                pdf_path = tmp.name

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(all_docs)

    @st.cache_resource(show_spinner=False)
    def get_vector_store(_splits):
        return Chroma.from_documents(
            _splits,
            embeddings,
            persist_directory="./chroma_index"
        )

    vector_store = get_vector_store(splits)
    retriever = vector_store.as_retriever()

    # History-aware retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful medical assistant. Given the chat history and the latest medical question, decide what to retrieve."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # QA prompt - restricted to medical domain
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional medical assistant. You only answer questions related to **medical science**. "
                   "If the question is not medical, respond with: '⚠️ I can only answer medical-related questions.'\n\n"
                   "Use the retrieved context to answer medically relevant questions. Be accurate and concise.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Chat history session state
    if "chat-history" not in st.session_state:
        st.session_state.chathistory = {}

    def get_history(session_id: str):
        if session_id not in st.session_state.chathistory:
            st.session_state.chathistory[session_id] = ChatMessageHistory()
        return st.session_state.chathistory[session_id]

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Chat Interface
    session_id = st.text_input("Enter a session ID:", type="default")
    user_question = st.chat_input("💬 Ask a medical question...")

    if user_question:
        history = get_history(session_id)
        result = conversational_rag.invoke({"input": user_question}, config={"configurable": {"session_id": session_id}})
        answer = result["answer"]

        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(answer)

        with st.expander("🕘 Full Chat History"):
            for msg in history.messages:
                role = getattr(msg, "role", msg.type)
                content = msg.content
                st.write(f"**{role.title()}:** {content}")
else:
    st.info("ℹ️ Please upload at least one medical-related PDF to begin.")