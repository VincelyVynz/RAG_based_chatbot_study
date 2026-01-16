import flet as ft
import requests
import threading
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from config import *


# ======================
# LOAD DATA
# ======================
with open("employee_data.txt", "r", encoding="utf-8") as f:
    employee_docs = [b.strip() for b in f.read().split("\n\n") if b.strip()]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedding_model.encode(employee_docs)

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings, dtype=np.float32))

conversation_history = []

# ======================
# STREAMING RAG
# ======================
def stream_rag_response(query, on_token):
    query_emb = embedding_model.encode([query])
    _, I = index.search(np.array(query_emb, dtype=np.float32), k=TOP_K)
    retrieved_docs = [employee_docs[i] for i in I[0]]

    history = ""
    for u, a in conversation_history[-MAX_HISTORY_TURNS:]:
        history += f"User: {u}\nAssistant: {a}\n"

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Conversation History:\n{history}\n"
        f"Context:\n" + "\n\n".join(retrieved_docs) +
        f"\n\nUser: {query}\nAssistant:"
    )

    with requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
        stream=True,
        timeout=120,
    ) as r:
        for line in r.iter_lines():
            if line:
                token = json.loads(line.decode()).get("response", "")
                if token:
                    on_token(token)

# ======================
# FLET APP
# ======================
def main(page: ft.Page):
    page.title = "RAG Chatbot"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = ft.Colors.BLACK
    page.padding = 0

    chat = ft.Column(
        expand=True,
        auto_scroll=True,
        scroll=ft.ScrollMode.AUTO,
        spacing=10,
    )

    def bubble(text: str, is_user: bool):
        text_widget = ft.Text(
            value=text,
            selectable=True,
            text_align=ft.TextAlign.RIGHT if is_user else ft.TextAlign.LEFT,
        )

        bubble_container = ft.Container(
            content=text_widget,
            padding=12,
            bgcolor=ft.Colors.BLUE_600 if is_user else ft.Colors.GREY_800,
            border_radius=18,
        )

        return ft.Row(
            controls=[
                ft.Container(
                    content=bubble_container,
                    alignment=ft.Alignment(1.0, 0.0) if is_user else ft.Alignment(-1.0, 0.0),
                    expand=True
                )
            ],
        )

    def send(_):
        user_msg = input_field.value.strip()
        if not user_msg:
            return

        input_field.disabled = True
        send_button.disabled = True
        input_field.value = ""
        page.update()

        chat.controls.append(bubble(user_msg, True))

        bot_row = bubble("", False)
        # Access the Text widget: Row -> Container(outer) -> Container(inner) -> Text
        bot_text_widget = bot_row.controls[0].content.content
        chat.controls.append(bot_row)
        page.update()

        def worker():
            full_reply = ""

            def on_token(tok):
                nonlocal full_reply
                full_reply += tok
                bot_text_widget.value = full_reply
                page.update()

            try:
                stream_rag_response(user_msg, on_token)
            except Exception as e:
                bot_text_widget.value = f"Error: {e}"

            conversation_history.append((user_msg, full_reply))
            if len(conversation_history) > MAX_HISTORY_TURNS:
                conversation_history.pop(0)

            input_field.disabled = False
            send_button.disabled = False
            page.update()

        threading.Thread(target=worker, daemon=True).start()

    input_field = ft.TextField(
        hint_text="Type a message",
        expand=True,
        filled=True,
        border_radius=24,
        on_submit=send,
    )

    send_button = ft.IconButton(
        icon=ft.Icons.SEND,
        on_click=send,
    )

    page.add(
        ft.Column(
            [
                ft.Container(chat, expand=True, padding=12),
                ft.Container(
                    ft.Row([input_field, send_button]),
                    padding=12,
                    bgcolor=ft.Colors.GREY_900,
                ),
            ],
            expand=True,
        )
    )

# ======================
# ENTRY POINT
# ======================
ft.run(main)
