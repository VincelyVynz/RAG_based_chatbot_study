# Step 1: Import libraries
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 2: Load employee data from text file
with open("employees.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into individual employee documents (blocks separated by two newlines)
employee_docs = [block.strip() for block in text.split("\n\n") if block.strip()]

print(f"Total employee documents loaded: {len(employee_docs)}\n")
print("Example employee document:\n", employee_docs[0], "\n")

# Step 3: Load sentence embedding model
# This model converts text into numeric vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 4: Convert employee documents into embeddings
embeddings = model.encode(employee_docs)
print("Embeddings created. Shape:", embeddings.shape, "\n")  # Should be (30, 384)

# Step 5: Create FAISS index for similarity search
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings, dtype=np.float32))
print("FAISS index created with", index.ntotal, "entries\n")


# Step 6: Query function
def search_employee(query, top_k=3):
    """Search for top_k most relevant employee documents."""
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb, dtype=np.float32), k=top_k)

    print(f"Query: {query}\n")
    print(f"Top {top_k} matches:\n")
    for i, idx in enumerate(I[0]):
        print(f"{i + 1}. {employee_docs[idx]}\n")


# Step 7: Example queries
search_employee("Who is the HR manager?")
search_employee("Who works in AI research?")
search_employee("Who is responsible for payroll?")
