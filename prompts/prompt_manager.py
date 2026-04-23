import os
import yaml
from pathlib import Path
from jinja2 import Template

class PromptManager:
    def __init__(self):
        """
        Initializes the manager by discovering and loading all per-agent YAML
        files located in the same directory as this module.

        Each YAML file must be named after the agent it configures
        (e.g. ``keyword_generator.yaml``). The file ``agents_config.yaml`` is
        intentionally excluded so that it can be kept as a reference/backup
        without interfering with the live configuration.
        """
        self._prompts_dir = Path(__file__).resolve().parent
        self._excluded_files = {"agents_config.yaml"}

        self._prompts: dict[str, dict] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> dict:
        """Parse a single YAML file and return its contents as a dict."""
        with open(path, "r", encoding="utf-8") as fh:
            try:
                data = yaml.safe_load(fh) or {}
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"[PromptManager] Syntax error in YAML file '{path}': {exc}"
                )
        return data

    def _load_all(self):
        """
        Scans the prompts directory for ``*.yaml`` files (excluding the legacy
        ``agents_config.yaml``) and registers each one under the agent name
        derived from the file stem (e.g. ``keyword_generator.yaml`` →
        ``'keyword_generator'``).
        """
        self._prompts.clear()

        yaml_files = sorted(self._prompts_dir.glob("*.yaml"))
        if not yaml_files:
            raise FileNotFoundError(
                f"[PromptManager] No YAML prompt files found in '{self._prompts_dir}'"
            )

        for yaml_path in yaml_files:
            if yaml_path.name in self._excluded_files:
                continue

            agent_name = yaml_path.stem  # e.g. "keyword_generator"
            self._prompts[agent_name] = self._load_file(yaml_path)

        if not self._prompts:
            raise FileNotFoundError(
                f"[PromptManager] No agent prompt files loaded from '{self._prompts_dir}'"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, agent_name: str, prompt_type: str, **kwargs) -> str:
        """
        Retrieves the requested Jinja2 template from the agent's YAML file and
        renders it with the provided variables.

        :param agent_name:  The name of the agent (e.g. ``'keyword_generator'``).
        :param prompt_type: The prompt key inside the agent file
                            (e.g. ``'system_prompt'``, ``'user_prompt'``).
        :param kwargs:      Variables to inject into the template.
        :return:            The rendered, stripped prompt string.
        """
        if agent_name not in self._prompts:
            raise KeyError(
                f"[PromptManager] Agent '{agent_name}' not found. "
                f"Available agents: {list(self._prompts.keys())}"
            )

        agent_config = self._prompts[agent_name]

        if prompt_type not in agent_config:
            raise KeyError(
                f"[PromptManager] Prompt type '{prompt_type}' missing for "
                f"agent '{agent_name}'. Available keys: {list(agent_config.keys())}"
            )

        template_str = agent_config[prompt_type]

        # Gracefully handle empty/null prompt entries
        if not template_str:
            return ""

        rendered = Template(template_str).render(**kwargs)
        return rendered.strip()

    def reload(self):
        """
        Reloads all per-agent YAML files from disk without restarting the
        application. Useful for hot-reloading prompt changes during development.
        """
        self._load_all()
        print(
            f"[PromptManager] All prompts successfully reloaded from "
            f"'{self._prompts_dir}'"
        )

    @property
    def agents(self) -> list[str]:
        """Returns the list of currently loaded agent names."""
        return list(self._prompts.keys())