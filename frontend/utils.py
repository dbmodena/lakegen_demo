import os
import asyncio
from typing import Dict, Any, Optional
import chainlit as cl
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
load_dotenv()

# Configuration constants
QUERY_ANALYZER_PROMPT = ""
# Load the prompt from a file
with open("backend/prompts/QueryAnalyzer.md", "r") as f:
    QUERY_ANALYZER_PROMPT = f.read()
    
TABLE_SELECTOR_PROMPT = ""
# Load the prompt from a file
with open("backend/prompts/TableSelector.md", "r") as f:
    TABLE_SELECTOR_PROMPT = f.read()
    
CODE_EXECUTOR_PROMPT = ""
# Load the prompt from a file
with open("backend/prompts/CoderCritic.md", "r") as f:
    CODE_EXECUTOR_PROMPT = f.read()



MODEL_SETTINGS = {
    "model": "Qwen/Qwen3-235B-A22B-fp8-tput",
    "temperature": 0.1,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "max_tokens": 10000,
}


DOWNLOAD_FOLDER = "downloads"
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)
    
MODEL_APIS = {
    "https://api.together.xyz/v1/": "TOGETHER_API_KEY",
    "https://api.openai.com/v1/": "OPENAI_API_KEY", 
    "https://api.mistral.ai/v1/": "MISTRAL_API_KEY"
}

MODEL_NAMES= {
    "https://api.together.xyz/v1/": "TOGETHER AI",
    "https://api.openai.com/v1/": "OpenAI",
    "https://api.mistral.ai/v1/": "MISTRAL AI"
}

PORTAL_CONFIGS = {
    "EU": "https://data.europa.eu/api/",
    "USA": "https://catalog.data.gov/api/",
    "CAN": "https://open.canada.ca/api/",
    "UK": "https://data.gov.uk/api/"
}

DEFAULT_SETTINGS = {
    "model_api": "https://api.together.xyz/v1/",
    "talk": False,
    "portal": "CAN",
    "top_k_results": 10,
    "n_keywords": 3
}

def get_analyzer_prompt(api_base_url: str, portal_url: str, top_k_results: int) -> str:
    formatted_prompt = QUERY_ANALYZER_PROMPT.format(
        api=api_base_url, portal=portal_url, top_k_results=top_k_results, n_keywords=3
    )
    #print(f"Formatted prompt: {formatted_prompt}")  # Debugging line
    return formatted_prompt

def get_model_client(settings: Dict[str, Any]) -> OpenAIChatCompletionClient:
    """Create and return an OpenAIChatCompletionClient based on settings."""
    model_api = settings.get("model_api", "https://api.together.xyz/v1/")
    api_key = os.getenv(MODEL_APIS.get(model_api))
    if not api_key:
        raise ValueError(f"API key for {model_api} not found in environment variables.")        
    return OpenAIChatCompletionClient(
        base_url=model_api,
        model="Qwen/Qwen3-235B-A22B-fp8-tput",
        api_key=api_key,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True
        },
        **{k: v for k, v in settings.items() if k not in ['model_api', 'portal', 'talk']}
    )



class SettingsValidator:
    """Validates chat settings and provides error messages."""
    
    @staticmethod
    def validate_model_api(api_url: str) -> tuple[bool, str]:
        """Validate model API URL."""
        if not api_url or api_url not in MODEL_APIS:
            return False, f"Invalid API URL. Must be one of: {list(MODEL_APIS.keys())}"
        return True, ""
    
    @staticmethod
    def validate_portal(portal: str) -> tuple[bool, str]:
        """Validate portal selection."""
        if not portal or portal not in PORTAL_CONFIGS:
            return False, f"Invalid portal. Must be one of: {list(PORTAL_CONFIGS.keys())}"
        return True, ""
    
    @staticmethod
    def validate_top_k_results(value: Any) -> tuple[bool, str]:
        """Validate top_k_results parameter."""
        try:
            k = int(value)
            if k <= 0:
                return False, "Top K results must be a positive integer"
            if k > 100:
                return False, "Top K results cannot exceed 100"
            return True, ""
        except (ValueError, TypeError):
            return False, "Top K results must be a valid integer"
    
    @staticmethod
    def validate_all_settings(settings: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate all settings and return any errors."""
        errors = []
        
        # Validate model API
        is_valid, error = SettingsValidator.validate_model_api(settings.get("model_api"))
        if not is_valid:
            errors.append(error)
        
        # Validate portal
        is_valid, error = SettingsValidator.validate_portal(settings.get("portal"))
        if not is_valid:
            errors.append(error)
        
        # Validate top_k_results
        is_valid, error = SettingsValidator.validate_top_k_results(settings.get("top_k_results"))
        if not is_valid:
            errors.append(error)
        
        return len(errors) == 0, errors

class APIKeyManager:
    """Manages API key validation and testing."""
    
    @staticmethod
    async def get_api_key_from_env(model_api: str) -> Optional[str]:
        """Get API key from environment variables."""
        env_key = MODEL_APIS.get(model_api)
        if not env_key:
            return None
        return os.getenv(env_key)
    
    @staticmethod
    async def prompt_for_api_key(model_api: str) -> Optional[str]:
        """Prompt user for API key if not found in environment."""
        try:
            res = await cl.AskUserMessage(
                content=f"🔑 Please provide your API key for {model_api}",
                timeout=30
            ).send()
            return res.get("output") if res else None
        except Exception as e:
            print(f"Error prompting for API key: {e}")
            return None
    
    @staticmethod
    async def validate_api_key(api_key: str, model_api: str, settings: Dict[str, Any]) -> bool:
        """Test if API key is valid by making a test request."""
        try:
            model_client = OpenAIChatCompletionClient(
                base_url=model_api,
                api_key=api_key,
                model_info={
                    "vision": False,
                    "function_calling": True,
                    "json_output": True,
                    "family": "unknown",
                    "structured_output": True
                },
                **{k: v for k, v in settings.items() if k not in ['model_api', 'portal', 'talk']}
            )
            
            # Test with a simple message
            await asyncio.wait_for(
                model_client.create([cl.Message(content="Test connection", source="user")]),
                timeout=10.0
            )
            return True
            
        except asyncio.TimeoutError:
            print("API key validation timed out")
            return False
        except Exception as e:
            print(f"API key validation failed: {e}")
            return False