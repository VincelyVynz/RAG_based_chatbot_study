from RAG_based_chatbot import *
import flet as ft

def main(page: ft.Page):
    page.title = "Employee RAG Chatbot"
    page.vertical_alignment = ft.MainAxisAlignment.START

    chat_output = ft.Column()

    user_input = ft.TextField(
        label = "Ask anything",
        autofocus = True,
        expand = True
    )

    send_button = ft.ElevatedButton("Send")


    def send_message(e):
        query = user_input.value.strip()
        if not query:
            return

        chat_output.controls.append(ft.Text(f"You: {query}", size = 16))
        page.update()

        response = search_docs(query)
        chat_output.controls.append(ft.Text(f"Assistant: {response}", size = 16))

        user_input.value = ""
        page.update()

    send_button.on_click = send_message
    user_input.on_submit = send_message

    page.add(
        chat_output,
        ft.Row(
            controls = [user_input, send_button]
        )
    )

ft.app(target=main)