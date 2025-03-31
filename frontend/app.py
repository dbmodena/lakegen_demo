import os

os.environ["no_proxy"] = "localhost,127.0.0.1,geonext.comune.modena.it"

import asyncio
import base64
import io
import json
import sqlite3
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Annotated, Dict

import autogen
import pandas as pd
import requests
import streamlit as st
from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
from autogen_agentchat.messages import TextMessage
from dotenv import load_dotenv
sys.path.append("backend")
from lakegen import main



# create logs

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "litellm.log")


# load_dotenv('/Users/angelomozzillo/tabgen/.env')
load_dotenv()

# setup page title and description
st.set_page_config(
    page_title="AutoGen Chat App with CKAN", page_icon="🤖", layout="wide"
)

if "form_submitted" not in st.session_state:
    st.session_state["form_submitted"] = False


# Define placeholders for model and key selection
selected_model = None
LOGO_IMAGE = "frontend/static/images/image.png"
# Sidebar for API key and model selection
st.markdown(
    """
    <style>
    .logo-container {
        display: flex;
        align-items: center;
    }
    .logo-img {
        width: 30px;  /* Adjust as needed */
        margin-right: 10px;
    }
    .logo-text {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.form(key="Form1"):
    with st.sidebar:
        logo_base64 = base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()
        st.markdown(
            f"""
            <div class="logo-container">
                <img class="logo-img" src="data:image/png;base64,{logo_base64}">
                <span class="logo-text">LakeGen</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.header("Open Data Portal")
        portals = ["EU", "USA", "CAN", "UK"]
        selected_portal = st.segmented_control(
            "Select a portal", portals, default="CAN"
        )

        top_k_results = st.number_input(
            "Top K", value=None, placeholder="Type a number..."
        )

        st.header("Model Configuration")
        api_key = st.text_input("Enter your Groq API key", type="password")

        if api_key:
            os.environ["GROQ_API_KEY"] = api_key

            # Start LiteLLM if not already running
            if "litellm_process" not in st.session_state:
                with open(log_file, "w") as log:
                    st.session_state.litellm_process = subprocess.Popen(
                        [
                            "litellm",
                            "--config",
                            "./config/config.yaml",
                            "--port",
                            "4323",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                st.success("LiteLLM started successfully!")

        selected_model = st.selectbox(
            "Model",
            [
                "llama-3.3-70b-versatile",
                "deepseek-r1-distill-llama-70b",
                "mixtral-8x7b-32768",
            ],
            index=0,
        )

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state["form_submitted"] = True


dict_portal = {
    "EU": "https://data.europa.eu/api/hub/search/ckan/package_search?q=",
    "USA": "https://catalog.data.gov/api/3/action/package_search?q=",
    "CAN": "https://open.canada.ca/data/api/3/action/package_search?q=",
    "UK": "https://data.gov.uk/api/action/package_search?q=",
}
# selected_key = st.text_input("GROQ API Key", type="password")

# Ensure valid API key and model are selected
# if not selected_key or not selected_model:
#   st.warning(
#      "Please provide a valid OpenAI API key and choose a preferred model", icon="⚠️"
# )
# st.stop()

# if the form is submitted

# Configure the OpenAI model
llm_config = {
    "config_list": [
        {
            "model": selected_model,
            "api_key": api_key,
            "api_type": "groq",
            "max_tokens": 7000,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.2,
            "temperature": 0,
            "seed": 42,
        },
    ],
    "temperature": 0,  # temperature of 0 means deterministic output
    "cache_seed": None,
    "seed": 42,
}

# CKAN API endpoint
path_api = dict_portal[selected_portal]


# Create a temporary directory to store the code files.
# temp_dir = tempfile.TemporaryDirectory()
from pathlib import Path

st.markdown(
    """
    🤖 **This is a demo of LakeGen using AutoGen agents with CKAN integration.**  
    It can fetch and analyze datasets from public CKAN APIs, process queries, and return relevant data. 📊  

    🔍 **Example query:**  
    *'What is the average salary of teachers with the most working days in Canada?'*  
    """
)


# Store chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# Main app container
with st.container():
    col2, col3 = st.columns([2, 2])
    with col2:
        mode = st.segmented_control(
            "Select a mode", ["Auto", "Explain", "Interactive"], default="Auto"
        )
    with col3:
        code = st.segmented_control(
            "Select a code", ["Python", "SQL"], default="Python"
        )

with st.container():
    # Create selectbox
    options = [
        "What is the average salary of teachers with the most working days in Canada?",
        "Another option...",
    ]

    selection = st.selectbox(
        "Select a question or enter your query:",
        options=options,
        index=None,
    )

    # Create text input for user entry
    if selection == "Another option...":
        user_query = st.text_input("Enter your query:")

    # Just to show the selected option
    if selection != "Another option...":
        user_query = selection

if st.session_state["form_submitted"]:
    with st.container():

        async def initiate_group_chat(
            user_query, path_api, selected_model, top_k_results
        ):
            if (
                "chat_history" not in st.session_state
                or st.session_state.last_query != user_query
            ):
                with st.spinner("Wait for it...", show_time=True):
                    chat = await main(
                        user_query, path_api, selected_model, top_k_results
                    )

                    if chat:
                        st.session_state.chat_history = chat  # Store chat history
                        st.session_state.last_query = user_query  # Remember last query
                    else:
                        st.session_state.chat_history = []  # Handle empty responses

        async def main_async_flow(user_query, path_api):
            await initiate_group_chat(
                user_query, path_api, selected_model, top_k_results
            )
            st.session_state.chat_initiated = True

        if user_query:
            if (
                "chat_history" not in st.session_state
                or st.session_state.last_query != user_query
            ):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(main_async_flow(user_query, path_api))
                loop.close()

        # Display chat without re-running the API call
        chat = st.session_state.get("chat_history", [])

        if chat:
            if mode == "Explain":
                for response in chat:
                    if isinstance(
                        response, TextMessage
                    ) and response.content.lower() not in [
                        "stop",
                        "exit",
                        "quit",
                        "approve",
                    ]:
                        try:
                            response.content = json.loads(
                                response.content
                            )  # Attempt JSON parsing
                        except:
                            pass

                        with st.chat_message(response.source, avatar="🤖"):
                            if isinstance(response.content, dict):
                                for i, result in enumerate(response.content):
                                    cleaned_result = (
                                        str(result)
                                        .replace("{", "")
                                        .replace("}", "")
                                        .replace("'", '"')
                                    )
                                    st.write(f"{i + 1}. {cleaned_result}")
                            else:
                                st.markdown(response.content)
            else:
                last_message = (
                    chat[-1].messages[-1].content
                    if chat and chat[-1].messages
                    else "No response."
                )
                with st.chat_message("LakeGen Assistant", avatar="🤖"):
                    st.markdown(last_message)


else:
    st.warning("Please submit the form before starting the chat.", icon="⚠️")
