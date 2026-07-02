import os
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

def build_and_save_vector_db():
    # 1. Use a FREE local model from Hugging Face instead of OpenAI
    print("⏳ Loading free local embedding model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 2. Initialize the local ChromaDB storage
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("interview_rag_project")
    
    # 3. Connect Chroma to LlamaIndex
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 4. Load the PDF files
    print("⏳ Processing and chunking documents inside /data...")
    documents = SimpleDirectoryReader("./data").load_data()
    
    # 5. Build and save the index locally
    print("🧠 Generating mathematical vectors locally and saving to ChromaDB...")
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print("✅ Success! Database generated and saved to ./chroma_db")

if __name__ == "__main__":
    build_and_save_vector_db()