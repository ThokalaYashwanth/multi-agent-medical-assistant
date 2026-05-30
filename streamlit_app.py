import streamlit as st
import requests

API_URL = "http://localhost:8000/api/v1/query"

st.set_page_config(page_title="Medical Assistant", page_icon="🏥", layout="wide")
st.title("🏥 AI-Powered Multi-Agent Medical Assistant")
st.caption("Powered by LangGraph · RAG · FastAPI — For informational purposes only")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        "This assistant uses specialized AI agents to handle:\n"
        "- 🔬 Symptom analysis & diagnosis\n"
        "- 📚 Medical literature (RAG)\n"
        "- 💊 Drug interactions\n"
        "- 📋 SOAP note generation"
    )
    if st.button("Clear chat"):
        st.session_state.messages = []

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Describe symptoms, ask a medical question, or request a SOAP note..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting specialist agents..."):
            try:
                resp = requests.post(
                    API_URL,
                    json={"query": prompt, "history": st.session_state.messages[:-1]},
                    timeout=30,
                )
                data = resp.json()
                response = data.get("response", "Error: no response")

                st.markdown(response)

                # Metadata badges
                col1, col2, col3 = st.columns(3)
                col1.metric("Intent", data.get("intent", "—").upper())
                col2.metric("Confidence", f"{data.get('confidence', 0):.0%}" if data.get("confidence") else "—")
                col3.metric("Review needed", "⚠️ Yes" if data.get("requires_human_review") else "✅ No")

                if data.get("safety_flags"):
                    st.warning(f"Safety flags: {', '.join(data['safety_flags'])}")

            except requests.exceptions.ConnectionError:
                response = "⚠️ Cannot connect to the API. Make sure the FastAPI server is running."
                st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
