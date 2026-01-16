
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"
# qwen2.5:1.5b
# phi3:mini
TOP_K = 30
MAX_HISTORY_TURNS = 3

SYSTEM_PROMPT = (
    "You are a helpful internal assistant. "
    "Answer only using the provided context and conversation history. "
    "If the answer is not in the context, either generate a normal response or if it is technical/facutal that you do not know, just say you do not know. "
    "Keep responses concise."
)