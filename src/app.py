import streamlit as st
from rag import ask_question


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="ShopEase AI Assistant",
    page_icon="🤖",
    layout="centered"
)


# ==================================================
# Application Title
# ==================================================

st.title("🤖 ShopEase AI Assistant")

st.write(
    "Ask questions about ShopEase's policies, products, "
    "shipping, sellers, and employee handbook."
)


# ==================================================
# User Input
# ==================================================

question = st.text_input(
    "Ask your question:",
    placeholder="Example: What is the return window for Smartphone A1?"
)


# ==================================================
# Ask Question
# ==================================================

if st.button("Ask"):

    if question.strip():

        with st.spinner("Searching documents and generating answer..."):

            answer = ask_question(question)

        st.subheader("Answer")

        st.write(answer)

    else:

        st.warning("Please enter a question.")