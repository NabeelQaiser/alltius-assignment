import streamlit as st
from chatbot_backend.agents.customer_support import answer_question
from langchain.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(page_title="Customer Support RAG Chatbot", page_icon="🤖")
st.title("Customer Support RAG Chatbot 🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thoughts" not in st.session_state:
    st.session_state.thoughts = []

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and i < len(st.session_state.thoughts):
            with st.expander("Show thinking steps", expanded=False):
                st.markdown(st.session_state.thoughts[i])

if prompt := st.chat_input("Ask a question about insurance or AngelOne support..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        cb_handler = StreamlitCallbackHandler(st.container())
        with st.spinner("Thinking..."):
            response = answer_question(prompt)
            # If response is a dict, extract the answer string
            if isinstance(response, dict):
                answer = response.get("result") or response.get("answer") or str(response)
            else:
                answer = response
            st.markdown(answer)
            # Optionally, you can log the callback handler's thoughts if available
            thoughts = getattr(cb_handler, "get_logs", lambda: "")()
            st.session_state.thoughts.append(thoughts)
        st.session_state.messages.append({"role": "assistant", "content": answer})
