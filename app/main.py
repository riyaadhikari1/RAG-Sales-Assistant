from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import pandas as pd
import os
import tempfile

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY not found in .env file")
    exit(1)
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize FastAPI
app = FastAPI(title="RAG Sales Assistant API")

# Single session storage
session_data = {
    "chat_history": [],
    "vectorstore": None,
    "qa_system": None,
    "documents_loaded": False
}

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    category: str

# Helper functions
def load_documents_from_files(uploaded_files: List[UploadFile]) -> List[Document]:
    all_docs = []
    
    for uploaded_file in uploaded_files:
        content = uploaded_file.file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.filename}") as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        if uploaded_file.filename.endswith('.csv'):
            df = pd.read_csv(tmp_path)
        elif uploaded_file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(tmp_path)
        else:
            os.unlink(tmp_path)
            continue
        
        for idx, row in df.iterrows():
            row_text = f"File: {uploaded_file.filename}\n"
            row_text += "\n".join([f"{col}: {row[col]}" for col in df.columns])
            
            doc = Document(
                page_content=row_text,
                metadata={"source": uploaded_file.filename, "row": idx}
            )
            all_docs.append(doc)
        
        os.unlink(tmp_path)
    
    return all_docs

def create_vectorstore(documents: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    
    temp_dir = tempfile.mkdtemp()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=temp_dir,
        collection_name="sales_documents"
    )
    
    return vectorstore

def create_question_classifier():
    classifier_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0
    )
    
    classifier_prompt = ChatPromptTemplate.from_messages([
        ('system', '''You are a question classifier. Analyze the user's question and classify it into ONE of these categories:

- ANALYTICS: Questions requiring calculations, aggregations, totals, sums, averages, counts
- COMPARISON: Questions comparing products, regions, time periods
- SUMMARY: Questions asking for overviews, lists, general information
- GENERAL: Simple lookup questions, specific data retrieval

Respond with ONLY the category name: ANALYTICS, COMPARISON, SUMMARY, or GENERAL'''),
        ('human', '{question}')
    ])
    
    chain = classifier_prompt | classifier_llm | StrOutputParser()
    
    def classify(question):
        result = chain.invoke({"question": question})
        return result.strip().upper()
    
    return classify

def create_specialized_chains(vectorstore, chat_history):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def get_chat_history():
        return chat_history[-6:] if chat_history else []
    
    # Analytics chain
    analytics_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.1
    )
    
    analytics_prompt = ChatPromptTemplate.from_messages([
        ('system', '''You are a data analytics expert. Your job is to perform calculations and aggregations.

Context:
{context}

When answering:
- Calculate sums, averages, counts, totals accurately
- Show your math/reasoning
- Format numbers clearly (e.g., $1,234.56)
- If data is incomplete, state what's missing'''),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{question}')
    ])
    
    analytics_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: get_chat_history()
        }
        | analytics_prompt
        | analytics_llm
        | StrOutputParser()
    )
    
    # Comparison chain
    comparison_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.2
    )
    
    comparison_prompt = ChatPromptTemplate.from_messages([
        ('system', '''You are a comparison analyst. Your job is to compare and rank items.

Context:
{context}

When answering:
- Compare products, regions, or time periods side-by-side
- Highlight key differences
- Rank items when appropriate (best to worst)
- Use clear formatting for comparisons'''),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{question}')
    ])
    
    comparison_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: get_chat_history()
        }
        | comparison_prompt
        | comparison_llm
        | StrOutputParser()
    )
    
    # Summary chain
    summary_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.4
    )
    
    summary_prompt = ChatPromptTemplate.from_messages([
        ('system', '''You are a business summarizer. Your job is to provide clear overviews.

Context:
{context}

When answering:
- Provide concise summaries
- List key items/products/regions
- Give high-level insights
- Keep it brief but informative'''),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{question}')
    ])
    
    summary_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: get_chat_history()
        }
        | summary_prompt
        | summary_llm
        | StrOutputParser()
    )
    
    # General chain
    general_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.3
    )
    
    general_prompt = ChatPromptTemplate.from_messages([
        ('system', '''You are a helpful sales assistant. Answer questions based on the context.

Context:
{context}

When answering:
- Be direct and precise
- If you don't know, say so
- Don't make up information'''),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{question}')
    ])
    
    general_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: get_chat_history()
        }
        | general_prompt
        | general_llm
        | StrOutputParser()
    )
    
    return {
        'analytics': analytics_chain,
        'comparison': comparison_chain,
        'summary': summary_chain,
        'general': general_chain
    }

def create_qa_system(vectorstore, chat_history):
    classifier = create_question_classifier()
    chains_dict = create_specialized_chains(vectorstore, chat_history)
    
    def ask_question(question: str):
        category = classifier(question)
        
        if category == "ANALYTICS":
            answer = chains_dict['analytics'].invoke(question)
        elif category == "COMPARISON":
            answer = chains_dict['comparison'].invoke(question)
        elif category == "SUMMARY":
            answer = chains_dict['summary'].invoke(question)
        else:
            answer = chains_dict['general'].invoke(question)
        
        return answer, category
    
    return ask_question

# API Endpoints

@app.get("/")
async def root():
    return {"status": "healthy", "message": "RAG Sales Assistant API"}

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    try:
        docs = load_documents_from_files(files)
        
        if not docs:
            raise HTTPException(status_code=400, detail="No documents loaded")
        
        session_data["vectorstore"] = create_vectorstore(docs)
        session_data["qa_system"] = create_qa_system(
            session_data["vectorstore"],
            session_data["chat_history"]
        )
        session_data["documents_loaded"] = True
        
        return {
            "message": f"Successfully loaded {len(docs)} documents",
            "document_count": len(docs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if not session_data["documents_loaded"]:
        raise HTTPException(status_code=400, detail="No documents loaded")
    
    try:
        answer, category = session_data["qa_system"](request.question)
        
        session_data["chat_history"].append(HumanMessage(content=request.question))
        session_data["chat_history"].append(AIMessage(content=answer))
        
        return QuestionResponse(answer=answer, category=category)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_chat_history():
    history = []
    for msg in session_data["chat_history"]:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})
    
    return {"history": history}

@app.delete("/history")
async def clear_chat_history():
    session_data["chat_history"] = []
    return {"message": "Chat history cleared"}

@app.get("/status")
async def get_status():
    return {
        "documents_loaded": session_data["documents_loaded"],
        "message_count": len(session_data["chat_history"])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)