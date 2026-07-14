import streamlit as st
from qa import get_answers

st.set_page_config(
    page_title="Funding Navigator AI",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Funding Navigator AI")
st.write(
    "Ask questions about Startup India funding schemes, policies, grants, and startup support."
)

question = st.text_input(
    "Enter your question:"
)

if st.button("Search"):

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:

        with st.spinner("Searching..."):

            result = get_answers([question])[0]

        st.subheader(" Answer")

        st.success(result["predicted_answer"])

        st.subheader("Retrieved Context")

        for i, chunk in enumerate(result["relevant_chunks"], start=1):

            with st.expander(
                f"Chunk {i} | Score: {chunk['score']:.3f}"
            ):
                st.write(chunk["text"])