A 100% offline, privacy-first Retrieval-Augmented Generation (RAG) system built using **LlamaIndex** and **ChromaDB**. This project parses, chunks, indexes, and queries unstructured documents completely inside your local hardware environment. No data ever leaks to external cloud APIs, and it incurs zero token billing overhead.

---

## 🏗️ Architectural Infrastructure

* **Data Ingestion & Orchestration Layer:** `LlamaIndex` core framework handling document parsing, token-based text partitioning, and vector mapping operations.
* **Persistent Vector Database:** `ChromaDB` local server instance, decoupling document ingestion from inference pipelines.
* **Local Semantic Embeddings:** `BAAI/bge-small-en-v1.5` running locally via Hugging Face Transformers to translate structural syntax into mathematical multi-dimensional spaces.
* **Inference LLM Generation Layer:** Meta's `Llama 3.2:1b` served locally via `Ollama` engine using 4-bit quantization models.

---

## ⚙️ Structural Blueprint & Code Artifacts

### 1. Ingestion Engine (`ingest.py`)
Iterates dynamically through the `./data` directory via `SimpleDirectoryReader`, partitions raw unstructured data blocks, converts text data elements into structural numeric tensors, and physically saves them down to the `./chroma_db` directory.

### 2. Inference Interface Loop (`query_server.py`)
Bootstraps a cold-start query loop. It maps user runtime natural language input to the exact same local text-embedding context, executes a vectorized semantic lookup across the `./chroma_db` vector matrices ($K\text{-counters}=5$), extracts localized parameters, and passes it through the quantized local model to yield high-fidelity, deterministic answers.

---

## 🏃‍♂️ Complete Local Installation & Execution Steps

### 1. Setup Your Virtual Workspace
Initialize your virtual environment isolated setup within your corporate or host root workspace folder:
```cmd
python -m venv venv
venv\Scripts\activate
2. Deploy Local Core Engines
Download and run the local AI model provider engine on your computer host system:

Download the native installer via ollama.com.

Open a standard command shell prompt window and pull down the lightweight parameter set:

DOS
ollama run llama3.2:1b
3. Install Framework Bindings
Install the isolated pip blocks required to bridge your code structures with the local hardware libraries:

DOS
pip install llama-index chromadb llama-index-vector-stores-chroma llama-index-embeddings-huggingface llama-index-llms-ollama
4. Build Your Local Vector Indexes
Create a directory named data/ in your project root, add your target data files (PDFs, TXT, MD), and parse them to storage:

DOS
mkdir data
python ingest.py
5. Initialize User Query Loop Server
Execute your offline interactive knowledge assistant server loop:

DOS
python query_server.py
