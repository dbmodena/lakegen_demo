import os
import yaml
from pathlib import Path
from jinja2 import Template

class PromptManager:
    def __init__(self, filename: str = "agents_config.yaml"):
        """
        Initializes the manager by loading and validating the YAML file in memory.
        """
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent
        self.filepath = project_root / "prompts" / filename

        self._load_prompts()

    def _load_prompts(self):
        """Loads the YAML file. Raises clear exceptions if something goes wrong."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"[PromptManager] File not found: {self.filepath}")
            
        with open(self.filepath, 'r', encoding='utf-8') as f:
            try:
                self.prompts = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise ValueError(f"[PromptManager] Syntax error in YAML file: {exc}")

    def render(self, agent_name: str, prompt_type: str, **kwargs) -> str:
        """
        Retrieves the requested YAML template and injects variables using Jinja2.
        
        param agent_name: The name of the agent (e.g., 'keyword_generator')
        param prompt_type: The type of prompt (e.g., 'system_prompt', 'user_prompt')
        param kwargs: The variables to inject (e.g., question="...", hint="...")
        return: The formatted and cleaned prompt string.
        """
        # Verify that the agent exists in the YAML file
        if agent_name not in self.prompts:
            raise KeyError(f"[PromptManager] Agent '{agent_name}' not configured in {self.filepath}")
            
        agent_config = self.prompts[agent_name]
        
        # Verify that the prompt type exists for that agent
        if prompt_type not in agent_config:
            raise KeyError(f"[PromptManager] Prompt type '{prompt_type}' missing for agent '{agent_name}'")
            
        template_str = agent_config[prompt_type]
        
        # Fallback handling if the prompt was left empty in the YAML
        if not template_str:
            return ""
        
        # Compile the Jinja2 template and inject variables
        template = Template(template_str)
        rendered_prompt = template.render(**kwargs)
        
        # Removes unnecessary whitespace or newlines at the beginning and end
        return rendered_prompt.strip()

    def reload(self):
        """
        Reload prompts at runtime without restarting the entire server or application.
        """
        self._load_prompts()
        print(f"[PromptManager] Prompts successfully reloaded from {self.filepath}")