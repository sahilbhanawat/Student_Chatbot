import streamlit as st
# We import the function from your backend file!
from RAG_chatbot import get_rag_chain

# --- UI Configuration ---
st.set_page_config(page_title="Student AI Tutor", page_icon="🎓")

st.title("🎓 Student Assistance Chatbot")
st.caption("🚀 Powered by Gemini & RAG | Separation of Concerns Edition")

# --- Initialize Backend ---
# @st.cache_resource ensures we only load the database ONCE, not every time you type.
@st.cache_resource
def load_chain():
    return get_rag_chain()

# Load the brain
rag_chain = load_chain()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I am ready to help with your studies."}]

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. Generate response using the backend chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if rag_chain:
                    response = rag_chain.invoke(prompt)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("The backend chain failed to load. Check your API key or Database path.")
            except Exception as e:
                st.error(f"Error: {e}")