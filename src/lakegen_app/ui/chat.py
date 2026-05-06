import streamlit as st


ASSISTANT_AVATAR = "static/favicon.png"


def chat_message(role: str):
    if role == "assistant":
        return st.chat_message(role, avatar=ASSISTANT_AVATAR)
    return st.chat_message(role)
