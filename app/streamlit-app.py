import streamlit as st
import requests
import os

st.set_page_config(page_title="RAG Sales Assistant", layout="wide")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

def upload_documents(files):
    try:
        files_data = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        response = requests.post(f"{API_BASE_URL}/upload", files=files_data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def ask_question(question):
    try:
        response = requests.post(f"{API_BASE_URL}/ask", json={"question": question})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def clear_history():
    try:
        response = requests.delete(f"{API_BASE_URL}/history")
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

st.title("RAG Sales Assistant")
st.markdown("Upload your sales data and ask questions")

with st.sidebar:
    st.header("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Documents", type="primary"):
        with st.spinner("Processing..."):
            result = upload_documents(uploaded_files)
            if result:
                st.session_state.documents_loaded = True
                st.success(result["message"])
    
    if st.session_state.documents_loaded:
        st.success("System Ready")
        
        if st.button("Clear Chat History"):
            if clear_history():
                st.session_state.chat_history = []
                st.success("History cleared")
                st.rerun()
    
    st.divider()
    st.markdown("**Example Questions**")
    st.markdown("""
    - What is the total sales revenue?
    - Which region has the highest sales?
    - Compare North vs South region
    - Show me all products
    """)

if not st.session_state.documents_loaded:
    st.info("Upload documents from the sidebar to get started")
else:
    for message in st.session_state.chat_history:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        with st.chat_message(role):
            st.write(content)
    
    if prompt := st.chat_input("Ask a question about your sales data..."):
        with st.chat_message("user"):
            st.write(prompt)
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_question(prompt)
                
                if result:
                    st.caption(f"Routed to: {result['category']}")
                    st.write(result['answer'])
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result['answer']
                    })

st.divider()
st.caption(f"API: {API_BASE_URL}")