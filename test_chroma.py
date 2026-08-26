import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

DB_DIR = './chroma_db'
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("Connecting to Vector DB...")
_chroma_settings = chromadb.Settings(
    chroma_server_thread_pool_size=8,
    anonymized_telemetry=False,
)
vector_db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
    client_settings=_chroma_settings,
)
print("Connected successfully! Vector DB loaded.")
print("Collection count:", vector_db._collection.count())
