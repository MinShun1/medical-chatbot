import streamlit as st
from src.chatbot import MedicalChatbot


# Page Configuration
st.set_page_config(
    page_title="Medical Document RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load RAG System
@st.cache_resource
def load_bot():
    return MedicalChatbot(
        api_key=st.secrets["GEMINI_API_KEY"],
        index_path="index/medical_index.faiss",
        metadata_path="index/metadata.pkl"
    )


bot = load_bot()


# Sidebar
with st.sidebar:

    st.title("Medical RAG Assistant")
    st.caption("Retrieval-Augmented Generation for Medical Documents")

    st.divider()

    st.subheader("System Overview")

    st.markdown("""
**Document Processing**
- Tesseract OCR
- spaCy & Regex

**Embedding Model**
- Gemini Embedding 001

**Vector Database**
- FAISS

**Language Model**
- Gemini 3.5 Flash Lite

**Knowledge Base**
- 1,000 medical records
""")

    st.divider()

    st.subheader("Example Queries")

    examples = [
        "Who is Amit Singh?",
        "Summarize med_doc_bill_100001_noisy.jpg",
        "List medications for Amit Singh",
        "What is Hypertension?",
        "What foods are recommended for hypertension?"
    ]

    for question in examples:
        st.caption(f"• {question}")

    st.divider()

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# Main Interface
st.title("Medical Document RAG Assistant")

st.markdown(
    """
Interact with a retrieval-augmented AI system to query information
from medical records and obtain responses based on retrieved documents.
"""
)

st.divider()


# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User Input
prompt = st.chat_input(
    "Enter your question about the medical records..."
)


if prompt:

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Retrieving relevant documents and generating response..."):

            answer = bot.ask(prompt)

        st.markdown(answer)

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# Footer
st.divider()

st.caption(
    "Medical Document RAG Assistant • "
    "For educational and research purposes only. "
    "This system does not provide medical diagnosis or treatment."
)
