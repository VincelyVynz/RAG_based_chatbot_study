import flet as ft
from RAG_based_chatbot import search_docs
import threading

def main(page: ft.Page):
    # Page settings
    page.title = "RAG Chatbot"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.window_min_width = 400
    page.window_min_height = 500

    # Chat messages container
    chat_column = ft.Column(
        spacing=12,
        expand=True,
        scroll=ft.ScrollMode.AUTO,
        auto_scroll=True, # Auto-scroll to bottom
    )

    # Message bubble factory
    def message_bubble(text, is_user=False):
        alignment = ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START
        bgcolor = "#7C4DFF" if is_user else "#2A2A2A"
        text_color = "white"
        
        # Calculate max width for bubble
        bubble_width = 300 # Fixed width or percentage could be used if needed

        return ft.Row(
            alignment=alignment,
            controls=[
                ft.Container(
                    padding=ft.padding.all(12),
                    bgcolor=bgcolor,
                    border_radius=ft.border_radius.only(
                        top_left=20, top_right=20, 
                        bottom_left=0 if is_user else 20, 
                        bottom_right=20 if is_user else 0
                    ),
                    content=ft.Text(
                        text,
                        color=text_color,
                        selectable=True,
                        size=14,
                    ),
                    constraints=ft.BoxConstraints(max_width=400), # Limit width
                )
            ]
        )

    # User input field
    user_input = ft.TextField(
        hint_text="Type your message...",
        expand=True,
        border_radius=20,
        filled=True,
        bgcolor="#1E1E1E",
        border_color="transparent",
        text_size=14,
        on_submit=lambda e: send_message_click(e),
    )

    # Send button
    send_button = ft.IconButton(
        icon=ft.icons.SEND,
        icon_color="white",
        bgcolor="#7C4DFF",
        tooltip="Send",
        on_click=lambda e: send_message_click(e),
    )

    # Bottom input bar
    input_bar = ft.Container(
        padding=10,
        bgcolor="#121212",
        content=ft.Row(
            controls=[
                user_input,
                send_button,
            ],
        ),
    )

    def get_response_async(user_text, loading_msg_control):
        # Get RAG response
        try:
            response = search_docs(user_text)
        except Exception as ex:
            response = f"Error: {ex}"

        # Update UI
        loading_msg_control.content.value = response
        page.update()

    # Send message handler
    def send_message_click(e):
        user_text = user_input.value.strip()
        if not user_text:
            return

        user_input.value = ""
        user_input.focus()

        # Show user message
        chat_column.controls.append(message_bubble(user_text, is_user=True))
        page.update()

        # Placeholder assistant message
        loading_text = ft.Text("Thinking...", color="white", size=14)
        loading_container = ft.Container(
            padding=ft.padding.all(12),
            bgcolor="#2A2A2A",
            border_radius=ft.border_radius.only(
                top_left=20, top_right=20, 
                bottom_left=20, bottom_right=0
            ),
            content=loading_text,
            constraints=ft.BoxConstraints(max_width=400),
        )
        
        loading_row = ft.Row(
            alignment=ft.MainAxisAlignment.START,
            controls=[loading_container]
        )
        
        chat_column.controls.append(loading_row)
        page.update()

        # Run search in a separate thread to avoid freezing UI
        # Note: Flet is thread-safe for page updates
        t = threading.Thread(target=get_response_async, args=(user_text, loading_container))
        t.start()

    # Layout
    page.add(
        ft.Column(
            expand=True,
            controls=[
                ft.Container(
                    expand=True,
                    padding=15,
                    content=chat_column,
                ),
                input_bar,
            ],
        )
    )

ft.app(target=main)
