import os
from functools import lru_cache

import tiktoken
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms.ollama import Ollama

from prompts.prompt_manager import PromptManager
from src.client_solr import LocalSolrClient


NUM_CTX = 32768

def get_llm(model, url) -> tuple[Ollama, TokenCountingHandler]:
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )
    Settings.callback_manager = CallbackManager([token_counter])
    
    llm = Ollama(
        model=model, 
        base_url=url, 
        request_timeout=300.0,
        temperature=0.1,
        context_window=NUM_CTX
    )

    return llm, token_counter


@lru_cache(maxsize=8)
def get_solr(core):
    return LocalSolrClient(core=core)


@lru_cache(maxsize=1)
def get_prompt_manager() -> PromptManager:
    return PromptManager()


def get_all_csv_files(csv_dir):
    try:
        return [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    except FileNotFoundError:
        return []

