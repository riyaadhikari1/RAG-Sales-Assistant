from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd
import os

all_docs = []

csv_files = [
       r'Product-Sales-Region.csv'   
]

for file in csv_files:
    try:
        df = pd.read_csv(file)
        
        for idx, row in df.iterrows():
            row_text = f"File: {file}\n"
            row_text += "\n".join([f"{col}: {row[col]}" for col in df.columns])
            
            doc = Document(
                page_content=row_text,
                metadata={
                    "source": file,
                    "row": idx,
                    "type": "csv"
                }
            )
            all_docs.append(doc)        
       
    except Exception as e:
        print(f"Error loading {file}: {e}")


# Excel files
excel_files = [
      r'Product-Sales-Region.xlsx'
]

for file in excel_files:
    try:
        df = pd.read_excel(file)
        
        for idx, row in df.iterrows():
            row_text = f"File: {file}\n"
            row_text += "\n".join([f"{col}: {row[col]}" for col in df.columns])
            
            doc = Document(
                page_content=row_text,
                metadata={
                    "source": file,
                    "row": idx,
                    "type": "excel"
                }
            )
            all_docs.append(doc)        
       
    except Exception as e:
        print(f"Error loading {file}: {e}")

print(f"Total documents loaded: {len(all_docs)}")

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],   
    chunk_size=500,      
    chunk_overlap=50
)

chunks = splitter.split_documents(all_docs)


# Load .env from parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Validate and export Google API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print(f"GOOGLE_API_KEY not found. Tried to load from: {dotenv_path}")
    print("Please add your GOOGLE_API_KEY to the .env file.")
    exit(1)
else:
    print(f"Google API key loaded")

# Export API key to environment so Google client picks it up
os.environ["GOOGLE_API_KEY"] = api_key

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)
 
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="sales_documents"
)



def create_question_classifier():
    
    classifier_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0
    )
    
    classifier_prompt = ChatPromptTemplate.from_messages([
        ('system', '''You are a question classifier. Analyze the user's question and classify it into ONE of these categories:

- ANALYTICS: Questions requiring calculations, aggregations, totals, sums, averages, counts
  Examples: "What is total revenue?", "How many products sold?", "Average sales per region?"

- COMPARISON: Questions comparing products, regions, time periods
  Examples: "Which region has higher sales?", "Compare product A vs B", "Best performing product?"

- SUMMARY: Questions asking for overviews, lists, general information
  Examples: "Show me all products", "What products do we have?", "Give me an overview"

- GENERAL: Simple lookup questions, specific data retrieval
  Examples: "What is the price of product X?", "Show sales in North region"

Respond with ONLY the category name: ANALYTICS, COMPARISON, SUMMARY, or GENERAL'''),
        ('human', '{question}')
    ])
    
    chain = classifier_prompt | classifier_llm | StrOutputParser()
    
    def classify(question):
        result = chain.invoke({"question": question})
        return result.strip().upper()
    
    return classify

def create_specialized_chains(vectorstore):
   
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 5}
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chat_history = []
    
    def get_chat_history():
        return chat_history[-6:] if chat_history else []
    
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
        'general': general_chain,
        'chat_history': chat_history
    }
    

def create_multi_chain_qa_system(vectorstore):
    
    classifier = create_question_classifier()
    chains_dict = create_specialized_chains(vectorstore)
    
    analytics_chain = chains_dict['analytics']
    comparison_chain = chains_dict['comparison']
    summary_chain = chains_dict['summary']
    general_chain = chains_dict['general']
    chat_history = chains_dict['chat_history']
    
    def ask_question(question):
        
        category = classifier(question)
        print(f"[Routing to: {category} chain]")
        
        if category == "ANALYTICS":
            answer = analytics_chain.invoke(question)
        elif category == "COMPARISON":
            answer = comparison_chain.invoke(question)
        elif category == "SUMMARY":
            answer = summary_chain.invoke(question)
        else: 
            answer = general_chain.invoke(question)
        
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        
        return answer
    
    return ask_question

qa_system = create_multi_chain_qa_system(vectorstore)

test_queries = [
    "What is the total sales revenue?", 
    "Which region has the highest sales?", 
    "What are the top selling products?", 
    "Show me sales data for the North region",
]

for query in test_queries:
    try:
        print(f"\n\nQ: {query}")
        result = qa_system(query)
        print(f"A: {result}")
    except Exception as e:
        print(f"\nError: {e}")

print("\nType 'quit' to exit\n")

# Interactive loop
while True:
    user_query = input("Your question: ")
    
    if user_query.lower() in ['quit', 'exit', 'q']:
        print("Exiting. Goodbye!")
        break
    
    if user_query.strip():
        try:
            print()
            result = qa_system(user_query)
            print(f"A: {result}\n")
        except Exception as e:
            print(f"\nError: {e}")
            