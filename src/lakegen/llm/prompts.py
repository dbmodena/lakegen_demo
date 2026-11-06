"""The agents' prompts: Markdown templates in ``prompts/`` at the repository root, rendered with Jinja2.

Each agent has a folder with one Markdown file per prompt:
``prompts/code_generator/system_prompt.md`` is the ``system_prompt`` of the
``code_generator`` agent. A file is a Jinja2 template; a ``{# ... #}`` comment
in it is a note for readers that the model never sees. ``LAKEGEN_PROMPTS_DIR``
points at another folder.
"""

from functools import lru_cache

from jinja2 import Environment, StrictUndefined, Template, TemplateSyntaxError

from lakegen.config.paths import prompts_dir


class PromptManager:
    """The Jinja2 prompt templates of every agent, compiled once from the prompts folder.

    Raises:
        ValueError: When a template does not compile.
        FileNotFoundError: When the folder holds no prompt file.
    """

    def __init__(self) -> None:
        folder = prompts_dir()
        # The templates render plain-text LLM prompts, not HTML: autoescaping
        # would rewrite characters such as < > & in the prompt text.
        environment = Environment(undefined=StrictUndefined, autoescape=False)  # nosec B701
        self._prompts: dict[str, dict[str, Template]] = {}
        for path in sorted(folder.glob("*/*.md")):
            try:
                template = environment.from_string(path.read_text(encoding="utf-8"))
            except TemplateSyntaxError as exc:
                raise ValueError(f"[PromptManager] {path}: {exc}") from exc
            self._prompts.setdefault(path.parent.name, {})[path.stem] = template
        if not self._prompts:
            raise FileNotFoundError(f"[PromptManager] No prompt files found in '{folder}'")

    def render(self, agent_name: str, prompt_type: str, **kwargs) -> str:
        """Render one of an agent's prompt templates with the given variables.

        Args:
            agent_name: The agent's folder, e.g. ``'code_generator'``.
            prompt_type: The prompt's file name without ``.md``, e.g. ``'system_prompt'``.
            **kwargs: Variables to inject into the template.

        Returns:
            The rendered prompt, stripped.

        Raises:
            KeyError: When the agent or its prompt type is unknown.
            RuntimeError: When the template fails to render, for instance
                because a variable is missing.
        """
        if agent_name not in self._prompts:
            raise KeyError(
                f"[PromptManager] Agent '{agent_name}' not found. "
                f"Available agents: {list(self._prompts)}"
            )
        agent_prompts = self._prompts[agent_name]
        if prompt_type not in agent_prompts:
            raise KeyError(
                f"[PromptManager] Prompt type '{prompt_type}' missing for "
                f"agent '{agent_name}'. Available keys: {list(agent_prompts)}"
            )
        try:
            rendered = agent_prompts[prompt_type].render(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"[PromptManager] Error rendering prompt '{prompt_type}' for agent '{agent_name}'. "
                f"Did you forget to pass a variable? Details: {exc}"
            ) from exc
        return rendered.strip()


@lru_cache(maxsize=1)
def get_prompt_manager() -> PromptManager:
    """The process-wide prompt manager, loaded once."""
    return PromptManager()
