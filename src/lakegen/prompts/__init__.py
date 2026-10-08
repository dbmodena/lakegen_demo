"""The prompt templates (one YAML file per agent) and the manager that renders them."""

from functools import lru_cache

from lakegen.prompts.prompt_manager import PromptManager


@lru_cache(maxsize=1)
def get_prompt_manager() -> PromptManager:
    return PromptManager()


__all__ = ["PromptManager", "get_prompt_manager"]
