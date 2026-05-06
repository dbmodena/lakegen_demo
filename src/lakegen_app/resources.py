import os
from typing import Any

import streamlit as st
import tiktoken
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms.ollama import Ollama

from lakegen_app.types import StreamCallback
from prompts.prompt_manager import PromptManager
from src.client_solr import LocalSolrClient


def get_llms(model, url, ctx) -> tuple[Ollama, Ollama, TokenCountingHandler]:
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )
    Settings.callback_manager = CallbackManager([token_counter])
    llm_versatile = Ollama(model=model, base_url=url, request_timeout=300.0,
                           temperature=0.1, additional_kwargs={"num_ctx": ctx})
    llm_instant = Ollama(model=model, base_url=url, request_timeout=300.0,
                         temperature=0.6,
                         additional_kwargs={"num_ctx": min(ctx, 8192),
                                            "presence_penalty": 0.1,
                                            "top_p": 0.7, "top_k": 30})
    return llm_versatile, llm_instant, token_counter


@st.cache_resource
def get_solr(core):
    return LocalSolrClient(core=core)


@st.cache_resource
def get_prompt_manager() -> PromptManager:
    return PromptManager()


def get_all_csv_files(csv_dir):
    try:
        return [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    except FileNotFoundError:
        return []


def make_streamlit_stream_callback(stream_placeholder: Any) -> StreamCallback:
    stream_text = ""

    def stream_to_placeholder(delta: str) -> None:
        nonlocal stream_text
        stream_text += delta
        stream_placeholder.markdown(stream_text)

    return stream_to_placeholder
