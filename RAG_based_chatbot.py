import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama endpoint for raw text generation

# CONSTANTS
OLLAMA_MODEL = "qwen2.5:1.5b"
TOP_K = 30

# Behaviour Prompt
SYSTEM_PROMPT = "You are a helpful internal assistant. Answer only using the provided context and conversation history. If the answer is not in the context, say you do not know. Keep responses concise."

conversation_history = []


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

with open("employee_data.txt", "r", encoding= "utf-8") as f:
    employee_docs = [block.strip() for block in f.read().split("\n\n") if block.strip()]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedding_model.encode(employee_docs)

dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings, dtype=np.float32))

def search_docs(query, top_k = TOP_K):
    query_emb = embedding_model.encode([query])
    D,I = index.search(np.array(query_emb, dtype = np.float32), k = top_k)
    retrieved_docs = [employee_docs[i] for i in I[0]]
    prompt = SYSTEM_PROMPT + "\n\nContext:\n" + "\n\n".join(retrieved_docs) + f"\n\nUser: {query}\nAssistant:"

    response = requests.post(OLLAMA_URL, json = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).json()
    return response.get("response", "").strip()

# Chat loop

print("RAG chatbot")
print("Ask a question about employee (type 'exit' to quit)\n")

while True:
    user_query = input("You: ").strip()

    if user_query.lower() in ["exit", "quit", "bye"]:
        print("\nChatbot: Goodbye! 👋")
        break

    if not user_query:
        print("Chatbot: Please enter a question.\n")
        continue

    assistant_response = search_docs(user_query)
    print(f"\nChatbot: {assistant_response}\n")