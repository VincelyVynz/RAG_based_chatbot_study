import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"
TOP_K = 30

SYSTEM_PROMPT = (
    "You are a helpful internal assistant. "
    "Answer only using the provided context and conversation history. "
    "If the answer is not in the context, say you do not know. "
    "Keep responses concise."
)

# =========================
# LOAD DATA & MODELS
# =========================

def load_documents(file_path="employee_data.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [block.strip() for block in f.read().split("\n\n") if block.strip()]
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Starting with empty knowledge base.")
        return []


employee_docs = load_documents()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

if employee_docs:
    doc_embeddings = embedding_model.encode(employee_docs)
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(doc_embeddings, dtype=np.float32))
else:
    index = None
    print("Index not initialized due to empty documents.")

# =========================
# RAG SEARCH FUNCTION
# =========================

def search_docs(query: str, top_k: int = TOP_K) -> str:
    """
    Searches relevant documents using FAISS and queries Ollama.
    Used by BOTH CLI and Flet UI.
    """
    if index is None:
        return "Error: No documents loaded in the knowledge base."

    try:
        query_emb = embedding_model.encode([query])
        distances, indices = index.search(
            np.array(query_emb, dtype=np.float32),
            k=min(top_k, len(employee_docs)),
        )

        retrieved_docs = [employee_docs[i] for i in indices[0]]

        prompt = (
            SYSTEM_PROMPT
            + "\n\nContext:\n"
            + "\n\n".join(retrieved_docs)
            + f"\n\nUser: {query}\nAssistant:"
        )

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=30,  # Reduced timeout
        )

        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Please ensure Ollama is running (e.g., 'ollama serve')."
    except requests.exceptions.Timeout:
        return "Error: Request to Ollama timed out. The model might be loading or the context is too large."
    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# CLI CHAT LOOP (OPTIONAL)
# =========================

def run_cli_chat():
    print("RAG chatbot")
    print("Ask a question about employees (type 'exit' to quit)\n")

    while True:
        user_query = input("You: ").strip()

        if user_query.lower() in {"exit", "quit", "bye"}:
            print("\nChatbot: Goodbye! 👋")
            break

        if not user_query:
            print("Chatbot: Please enter a question.\n")
            continue

        assistant_response = search_docs(user_query)
        print(f"\nChatbot: {assistant_response}\n")

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    run_cli_chat()
