"""
LakeGen Interactive - Chainlit application.
Run with: uv run chainlit run src/app.py -w
"""

import sys
import logging
from pathlib import Path

import chainlit as cl
import sniffio
from chainlit.server import app as chainlit_app
from chainlit.input_widget import Select, TextInput

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from lakegen.bootstrap import (  # noqa: E402
    bootstrap_nltk_data,
    ensure_project_paths,
    nltk_download_dir,
)
from lakegen.ui.state import (  # noqa: E402
    MODEL_OPTIONS,
    SOLR_CORE_OPTIONS,
    RuntimeSettings,
    get_session,
    set_runtime_settings,
)
from lakegen.ui.i18n import t  # noqa: E402
from lakegen.ui.workflow import run_lakegen_workflow  # noqa: E402

ensure_project_paths(_SRC_DIR, _ROOT_DIR)

logger = logging.getLogger(__name__)


class AsyncioSniffioMiddleware:
    """Provide AnyIO's async-backend context for Chainlit static routes."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        token = sniffio.current_async_library_cvar.set("asyncio")
        try:
            await self.app(scope, receive, send)
        finally:
            sniffio.current_async_library_cvar.reset(token)


chainlit_app.add_middleware(AsyncioSniffioMiddleware)


def _settings_widgets(runtime: RuntimeSettings | None = None) -> list:
    runtime = runtime or RuntimeSettings.default()
    return [
        TextInput(
            id="ollama_url",
            label=t("settings.ollama_url"),
            initial=runtime.ollama_url,
        ),
        Select(
            id="model_name",
            label=t("settings.model"),
            values=MODEL_OPTIONS,
            initial_value=runtime.model_name,
        ),
        Select(
            id="solr_core",
            label=t("settings.solr_core"),
            values=SOLR_CORE_OPTIONS,
            initial_value=runtime.solr_core,
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    try:
        bootstrap_error = bootstrap_nltk_data()
        if bootstrap_error:
            await cl.Message(
                content=(
                    f"{bootstrap_error}\n\n"
                    "Run:\n"
                    "```bash\n"
                    f"uv run python -m nltk.downloader -d {nltk_download_dir()} "
                    "wordnet omw-1.4 stopwords\n"
                    "```"
                )
            ).send()
            return

        session = get_session()

        runtime = RuntimeSettings.default()

        settings = await cl.ChatSettings(_settings_widgets(runtime)).send()
        runtime = RuntimeSettings.from_chainlit_settings(settings or {})
        set_runtime_settings(runtime)
        session.runtime = runtime

        await cl.Message(
            content=(
                f"{t('app.title')}\n\n"
                f"{t('app.intro')}"
            )
        ).send()
    except Exception as exc:
        logger.exception("LakeGen failed during on_chat_start")
        await cl.Message(
            content=f"LakeGen startup failed: `{type(exc).__name__}: {exc}`"
        ).send()
        raise


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    try:
        runtime = RuntimeSettings.from_chainlit_settings(settings or {})
        set_runtime_settings(runtime)
        get_session().runtime = runtime
        await cl.Message(
            content=t(
                "app.settings_updated",
                model_name=runtime.model_name,
                solr_core=runtime.solr_core,
            )
        ).send()
    except Exception as exc:
        logger.exception("LakeGen failed during on_settings_update")
        await cl.Message(
            content=f"LakeGen settings update failed: `{type(exc).__name__}: {exc}`"
        ).send()
        raise


@cl.on_message
async def on_message(message: cl.Message) -> None:
    try:
        await run_lakegen_workflow(message.content)
    except Exception as exc:
        logger.exception("LakeGen failed during on_message")
        await cl.Message(
            content=f"LakeGen workflow failed: `{type(exc).__name__}: {exc}`"
        ).send()
        raise
