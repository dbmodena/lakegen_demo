"""Non-interactive LakeGen workflow used by API and batch clients."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from lakegen.core.config import BASE_DIR, resolve_portal_tables_dir
from lakegen.core.logger import save_experiment_log
from lakegen.core.resources import get_all_table_files, get_llm, get_prompt_manager, get_solr
from lakegen.phases import phase12_agent, phase3_generate_and_execute, phase4_synthesize
from lakegen.ui.state import MODEL_OPTIONS, SOLR_CORE_OPTIONS, RuntimeSettings
from lakegen.retrieval import RetrievalConfig, RetrievalMode


MAX_CODE_ATTEMPTS = 3
MAX_TABLE_ATTEMPTS = 3
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuestionSource:
    """A question together with its location in the submitted JSON document."""

    question: str
    path: str
    source_id: str | int | None = None


@dataclass
class QueryResult:
    question: str
    status: str
    answer: str = ""
    raw_result: str = ""
    code: str = ""
    tables: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    tokens: dict[str, int] = field(
        default_factory=lambda: {"p1_p2": 0, "p3": 0, "p4": 0}
    )
    retries: int = 0
    error: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_path(parent: str, component: str | int) -> str:
    if isinstance(component, int):
        return f"{parent}[{component}]"
    if component.isidentifier():
        return f"{parent}.{component}"
    return f"{parent}[{component!r}]"


def extract_questions(payload: Any) -> list[QuestionSource]:
    """Extract questions from both simple lists and queries_old-style documents.

    Supported examples include ``["question"]``, ``{"questions": [...]}``, and
    the historical nested ``...data.queries[*].question`` structure. Occurrences
    are preserved deliberately: benchmark files may contain intentional repeats.
    """

    found: list[QuestionSource] = []

    def visit(value: Any, path: str, *, string_is_question: bool = False) -> None:
        if isinstance(value, str):
            question = value.strip()
            if string_is_question and question:
                found.append(QuestionSource(question=question, path=path))
            return

        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, _json_path(path, index), string_is_question=string_is_question)
            return

        if not isinstance(value, dict):
            return

        question = value.get("question")
        if isinstance(question, str) and question.strip():
            found.append(
                QuestionSource(
                    question=question.strip(),
                    path=_json_path(path, "question"),
                    source_id=value.get("id"),
                )
            )

        for key, item in value.items():
            if key == "question":
                continue
            visit(
                item,
                _json_path(path, str(key)),
                string_is_question=key in {"questions", "queries"},
            )

    visit(payload, "$", string_is_question=isinstance(payload, list))
    return found


def make_runtime_settings(
    *,
    core: str,
    model: str,
    use_unified_agent: bool = True,
    retrieval_mode: RetrievalMode | str = RetrievalMode.KEYWORD,
    top_k: int = 10,
    alpha: float = 0.5,
    candidate_multiplier: int = 5,
) -> RuntimeSettings:
    if core not in SOLR_CORE_OPTIONS:
        raise ValueError(
            f"Unknown core {core!r}. Expected one of: {', '.join(SOLR_CORE_OPTIONS)}"
        )
    if model not in MODEL_OPTIONS:
        raise ValueError(
            f"Unknown model {model!r}. Expected one of: {', '.join(MODEL_OPTIONS)}"
        )

    return RuntimeSettings(
        model_name=model,
        solr_core=core,
        csv_dir=resolve_portal_tables_dir(core),
        db_path=BASE_DIR / f"data/blend_{core}.db",
        use_unified_agent=use_unified_agent,
        retrieval=RetrievalConfig.from_env(
            mode=retrieval_mode,
            top_k=top_k,
            alpha=alpha,
            candidate_multiplier=candidate_multiplier,
        ),
    )


def run_question(question: str, runtime: RuntimeSettings) -> QueryResult:
    """Run the automatic unified LakeGen workflow for one question."""

    started = time.monotonic()
    result = QueryResult(question=question, status="running")
    llm, _token_counter = get_llm(runtime.model_name)
    solr = get_solr(runtime.solr_core)
    prompt_manager = get_prompt_manager()
    all_files = get_all_table_files(runtime.csv_dir)
    if not all_files:
        raise RuntimeError(f"No CSV or Parquet tables found in {runtime.csv_dir}")

    run_dir = BASE_DIR / "coding" / uuid.uuid4().hex
    selected: list[str] = []
    keywords: list[str] = []
    solr_meta: dict[str, dict[str, Any]] = {}
    reasoning = ""
    trace = ""
    hint = ""
    error = ""

    try:
        for table_attempt in range(MAX_TABLE_ATTEMPTS):
            (
                selected,
                keywords,
                solr_meta,
                reasoning,
                trace,
                architecture_tokens,
            ) = phase12_agent(
                query=question,
                llm=llm,
                pm=prompt_manager,
                all_files=all_files,
                solr_client=solr,
                csv_dir=runtime.csv_dir,
                hint=hint,
                portal_name=runtime.portal_name,
                retrieval_config=runtime.retrieval,
            )
            result.tokens["p1_p2"] += architecture_tokens
            result.tables = selected
            result.keywords = keywords

            previous_code = ""
            for code_attempt in range(MAX_CODE_ATTEMPTS):
                generated = phase3_generate_and_execute(
                    question,
                    selected,
                    selected,
                    solr_meta,
                    reasoning,
                    llm,
                    prompt_manager,
                    runtime.csv_dir,
                    retries=code_attempt,
                    error_msg=error,
                    previous_code=previous_code,
                    run_dir=run_dir,
                )
                result.tokens["p3"] += generated.tokens
                result.retries += int(code_attempt > 0)
                previous_code = generated.clean_code or generated.code_raw
                result.code = generated.clean_code

                if generated.rejected_reason:
                    hint = (
                        "The code generator rejected the previous tables. "
                        f"Select different tables. Feedback: {generated.rejected_reason}"
                    )
                    error = generated.rejected_reason
                    break

                if generated.error is None and generated.raw_result is not None:
                    result.raw_result = generated.raw_result
                    answer, synthesis_tokens = phase4_synthesize(
                        question, generated.raw_result, llm, prompt_manager
                    )
                    result.answer = answer
                    result.tokens["p4"] = synthesis_tokens
                    result.status = "completed"
                    result.error = ""
                    return result

                error = generated.error or "Code execution returned no output."
            else:
                break

            if table_attempt == MAX_TABLE_ATTEMPTS - 1:
                break

        result.status = "failed"
        result.error = error or "LakeGen could not produce an executable answer."
        return result
    except Exception as exc:
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        result.elapsed_seconds = round(time.monotonic() - started, 3)
        try:
            save_experiment_log(
                question=question,
                code=result.code,
                result=result.raw_result,
                retries=result.retries,
                reasoning=reasoning,
                tables=selected,
                final_keywords=keywords,
                final_result=result.answer,
                full_trace=trace,
                tokens_phase1=result.tokens["p1_p2"],
                tokens_phase3=result.tokens["p3"],
                tokens_phase4=result.tokens["p4"],
                error=result.error,
                csv_filename="api_experiments_log.csv",
                model=runtime.model_name,
                # The non-interactive API currently executes phase12_agent.
                architecture="unified",
            )
        except Exception:
            logger.exception("Could not persist the LakeGen experiment log")
