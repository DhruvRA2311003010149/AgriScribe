import os
import warnings
import json

# --- FIX: Set ChromaDB SQLite pool/timeout env vars BEFORE importing chromadb ---
os.environ["CHROMA_SQLITE_POOL_SIZE"] = "10"          # increase pool size
os.environ["CHROMA_SQLITE_TIMEOUT"] = "60"            # 60-second connection timeout
os.environ["CHROMA_SERVER_THREAD_POOL_SIZE"] = "8"    # more threads available

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI # For Query Analysis
from langchain_ollama import OllamaLLM # Keeping Llama 3 for the final answer
from langchain_core.prompts import PromptTemplate
# --- SECURITY ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyAf6nR_5HAQkAfvPp5PhxWydth_STwhm0Y"

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['HF_HUB_OFFLINE'] = '1'

# --- CONFIGURATION ---
DB_DIR = "./chroma_db"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 1. Initialize Multilingual Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. Connect to Vector DB
if os.path.exists(DB_DIR):
    # Use explicit Settings to increase pool size and timeouts (fixes macOS 'pool timed out')
    _chroma_settings = chromadb.Settings(
        chroma_server_thread_pool_size=8,
        anonymized_telemetry=False,
    )
    vector_db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        client_settings=_chroma_settings,
    )
else:
    vector_db = None

# 3. Initialize Models
# Gemini is used for the fast Query Analysis (Filtering logic)
analyzer_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
# Llama 3 remains your "Scientific Validator" for the final answer
final_llm = OllamaLLM(model="llama3", temperature=0)

# --- NOVELTY: THE QUERY ANALYZER (from the video) ---
ANALYZER_PROMPT = PromptTemplate.from_template("""
Analyze the user's agricultural question and extract search filters.
Return ONLY a JSON object with: "crop", "topic", "language", and "search_query".
If a filter is not mentioned, use "any".

USER QUESTION: {question}
JSON:
""")

def get_structured_filters(question):
    """Translates natural language into database filters."""
    try:
        response = analyzer_llm.invoke(ANALYZER_PROMPT.format(question=question))
        clean_json = response.content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except:
        return {"crop": "any", "topic": "any", "language": "any", "search_query": question}

def get_answer(question):
    if not vector_db: 
        return "❌ Database not found. Please run ingest.py first."

    # Step 1: Analyze Query (The Video's Method)
    filters = get_structured_filters(question)
    print(f"🛠️ Applied Filters: {filters}")

    # Step 2: Build the ChromaDB Metadata Filter
    where_clause = {}
    if filters["crop"] != "any": where_clause["crop"] = filters["crop"]
    if filters["language"] != "any": where_clause["language"] = filters["language"]
    
    # Step 3: Structured Retrieval
    # We search using the refined search_query and apply the hard metadata filter
    docs = vector_db.similarity_search(
        filters.get("search_query", question), 
        k=6, 
        filter=where_clause if where_clause else None
    )

    if not docs:
        return "⚠️ No relevant information found matching these specific criteria."

    # Combine context and track sources
    raw_context = ""
    sources = set()
    for doc in docs:
        # Include metadata in the context so Llama knows the specific tags
        raw_context += f"SOURCE: {doc.metadata.get('source')} | CROP: {doc.metadata.get('crop')}\nCONTENT: {doc.page_content}\n\n"
        sources.add(doc.metadata.get('source', 'Unknown'))

    # Step 4: Final Generation (Your Research Prompt)
    prompt_template = """
    SYSTEM: You are the AgriScribe Research Validator. You provide technical answers based EXCLUSIVELY on the provided agricultural documents.
    
    CONTEXT FROM PDFS:
    {context}

    USER QUESTION: {question}

    SCIENTIFIC VERIFICATION & ANSWER:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    response = final_llm.invoke(prompt.format(context=raw_context, question=question))
    
    source_block = "\n\n--- \n📚 **Verified Research Sources:**\n" + "\n".join([f"- {s}" for s in sorted(list(sources))])
    
    return response + source_block

if __name__ == "__main__":
    print("\n🌾 AgriScribe: Structured Research Engine (V5 - High Accuracy) 🌾")
    user_q = input("Enter query: ")
    if user_q.strip():
        print(get_answer(user_q))