import streamlit as st
from src.chatbot import MedicalChatbot

# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="Medical Document RAG Assistant",
    page_icon="🩺",
    layout="wide"
)

# ==========================
# Load chatbot
# ==========================

@st.cache_resource
def load_bot():
    return MedicalChatbot(
        api_key=st.secrets["GEMINI_API_KEY"],
        index_path="data/medical_index.faiss",
        metadata_path="data/metadata.pkl"
    )

bot = load_bot()

# ==========================
# Sidebar
# ==========================

with st.sidebar:

    st.title("🩺 Medical RAG")

    st.markdown("---")

    st.subheader("Project")

    st.write("""
**OCR**
- Tesseract

**LLM**
- Gemini 3.5 Flash Lite

**Embedding**
- Gemini Embedding 001

**Vector Database**
- FAISS

**Documents**
- 1000 Medical Records
""")

    st.markdown("---")

    st.subheader("Example Questions")

    examples = [
        "Who is Amit Singh?",
        "Summarize med_doc_bill_100001_noisy.jpg",
        "List medications for Amit Singh",
        "What is Hypertension?",
        "What foods are recommended for hypertension?"
    ]

    for q in examples:
        st.caption("• " + q)

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ==========================
# Header
# ==========================

st.title("🩺 Medical Document RAG Assistant")

st.write(
    "Ask questions about patient records or general medical information."
)

# ==========================
# Chat History
# ==========================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================
# Chat Input
# ==========================

prompt = st.chat_input(
    "Ask anything about the medical documents..."
)

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching medical records..."):

            answer = bot.ask(prompt)

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })