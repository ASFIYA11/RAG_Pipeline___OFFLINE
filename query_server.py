import sys
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

def ask_local_rag(query_text: str):
    # 1. Connect to your free, local LLM running via Ollama
    Settings.llm = Ollama(model="llama3.2:1b", request_timeout=60.0)
    
    # 2. Set the exact same embedding model we used to build the database
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 3. Open your local ChromaDB directory
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_collection("interview_rag_project")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 4. Load the mathematical index directly out of storage
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    
    # 5. Create the engine to fetch top matching chunks and generate the answer
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(query_text)
    return response

if __name__ == "__main__":
    print("\n🤖 Local, Offline RAG Engine Initialized.")
    print("👉 Type your question and hit Enter. Type 'exit' to quit.\n")
    
    while True:
        try:
            question = input("🧐 Ask something about your document: ")
            if question.lower() == 'exit':
                print("\nShutting down server. Good luck with your interview, bro! 🚀")
                break
            if not question.strip():
                continue
                
            print("🤖 AI is thinking...")
            answer = ask_local_rag(question)
            print(f"\n🤖 AI Response:\n{answer}\n")
            print("-" * 50)
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Make sure Ollama is running in the background with 'ollama run llama3.2:1b'\n")