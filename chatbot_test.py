from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

with open("employee_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

employee_docs = [block.strip() for block in text.split("\n\n") if block.strip()]

print(f"Total employee documents loaded: {len(employee_docs)}\n")

# This model converts text into numeric vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(employee_docs)
print("Embeddings created. Shape:", embeddings.shape, "\n")  # Should be (30, 384)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))
print("FAISS index created with", index.ntotal, "entries\n")


def search_employee(query, top_k=2):
    """Search for top_k most relevant employee documents."""
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb, dtype=np.float32), k=top_k)

    print(f"Query: {query}\n")
    print(f"Top {top_k} matches:\n")
    for i, idx in enumerate(I[0]):
        print(f"{i + 1}. {employee_docs[idx]}\n")


print("🤖 Employee Search Chatbot")
print("Ask a question about employees (type 'exit' to quit)\n")

while True:
    user_query = input("You: ").strip()

    if user_query.lower() in ["exit", "quit", "bye"]:
        print("\nChatbot: Goodbye! 👋")
        break

    if not user_query:
        print("Chatbot: Please enter a question.\n")
        continue

    search_employee(user_query)
