"""
LakeGen Interactive - Chainlit application.
Run with: uv run chainlit run src/app.py

Do not use Chainlit's --watch flag during normal runs: LakeGen writes generated
Python scripts under coding/, and a file watcher would restart the active chat.
"""

import sys
import asyncio
import logging
from pathlib import Path

import chainlit as cl
import sniffio
from chainlit.server import app as chainlit_app
from chainlit.input_widget import Select, Switch

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from lakegen.core.bootstrap import (  # noqa: E402
    bootstrap_nltk_data,
    ensure_project_paths,
    nltk_download_dir,
)
from lakegen.ui.state import (  # noqa: E402
    MODEL_OPTIONS,
    SOLR_CORE_OPTIONS,
    SOLR_CORE_PORTAL_NAMES,
    RuntimeSettings,
    LakeGenSession,
    WorkflowCancelled,
    get_runtime_settings,
    get_session,
    set_runtime_settings,
)
from lakegen.ui.i18n import t  # noqa: E402
from lakegen.ui.starters import starters_for_core  # noqa: E402
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
        Select(
            id="model_name",
            label=t("settings.model"),
            values=MODEL_OPTIONS,
            initial_value=runtime.model_name,
        ),
        Switch(
            id="use_unified_agent",
            label="Use Unified Agent (Phase 1 & 2)",
            initial=runtime.use_unified_agent,
        ),
    ]


def _selected_solr_core() -> str:
    chat_profile = str(cl.user_session.get("chat_profile") or "")
    if chat_profile in SOLR_CORE_OPTIONS:
        return chat_profile
    return RuntimeSettings.default().solr_core


@cl.set_chat_profiles  # type: ignore
async def chat_profiles():
    default_core = RuntimeSettings.default().solr_core
    return [
        cl.ChatProfile(
            name=core,
            display_name=SOLR_CORE_PORTAL_NAMES.get(core, core),
            markdown_description=(
                f"Ask questions against {SOLR_CORE_PORTAL_NAMES.get(core, core)}."
            ),
            default=core == default_core,
            starters=starters_for_core(core),
        )
        for core in SOLR_CORE_OPTIONS
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

        selected_solr_core = _selected_solr_core()
        runtime = RuntimeSettings.from_chainlit_settings(
            {},
            solr_core=selected_solr_core,
        )

        settings = await cl.ChatSettings(_settings_widgets(runtime)).send()
        runtime = RuntimeSettings.from_chainlit_settings(
            settings or {},
            solr_core=selected_solr_core,
        )
        set_runtime_settings(runtime)
        session.runtime = runtime
    except Exception as exc:
        logger.exception("LakeGen failed during on_chat_start")
        await cl.Message(
            content=f"LakeGen startup failed: `{type(exc).__name__}: {exc}`"
        ).send()
        raise


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    try:
        current_runtime = get_runtime_settings()
        runtime = RuntimeSettings.from_chainlit_settings(
            settings or {},
            solr_core=current_runtime.solr_core,
        )
        set_runtime_settings(runtime)
        get_session().runtime = runtime
        agent_mode = "Unified" if runtime.use_unified_agent else "Divided"
        await cl.Message(
            content=t(
                "app.settings_updated",
                model_name=runtime.model_name,
                solr_core=runtime.solr_core,
                agent_mode=agent_mode,
            )
        ).send()
    except Exception as exc:
        logger.exception("LakeGen failed during on_settings_update")
        await cl.Message(
            content=f"LakeGen settings update failed: `{type(exc).__name__}: {exc}`"
        ).send()
        raise


@cl.on_stop
async def on_stop() -> None:
    """Called when the user clicks the Stop button in the UI."""
    try:
        session = get_session()
        session.request_cancel()
        logger.info("User requested workflow cancellation.")
    except Exception:
        pass


@cl.on_message
async def on_message(message: cl.Message) -> None:
    try:
        old_session = get_session()
        new_session = LakeGenSession(
            runtime=old_session.runtime,
            query=message.content,
        )
        new_session.workflow_task = asyncio.current_task()
        cl.user_session.set("lakegen_session", new_session)
        
        await run_lakegen_workflow(message.content)
    except (asyncio.CancelledError, WorkflowCancelled):
        logger.info("Workflow cancelled by user.")
        await cl.Message(content="⏹ Workflow stopped.").send()
    except Exception as exc:
        logger.exception("LakeGen failed during on_message")
        await cl.Message(
            content=f"LakeGen workflow failed: `{type(exc).__name__}: {exc}`"
        ).send()
        raise
