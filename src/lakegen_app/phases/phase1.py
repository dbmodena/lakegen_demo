import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

from llama_index.core.llms import ChatMessage, LLM

from lakegen_app.phase2_logging import format_cli_log_value
from lakegen_app.types import SolrMetadata, StreamCallback
from prompts.prompt_manager import PromptManager
from src.client_solr import LocalSolrClient

from .utils import (
    emit_candidate_summary,
    match_local_csv,
    solr_metadata_from_doc,
)



def extract_wordnet_query_keywords(query: str) -> str:
    lemmatizer = WordNetLemmatizer()
    words = re.findall(r'\b\w+\b', query.lower())
    
    try:
        ita_stops = set(stopwords.words('italian'))
        spa_stops = set(stopwords.words('spanish'))
        fra_stops = set(stopwords.words('french'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        ita_stops = set(stopwords.words('italian'))
        spa_stops = set(stopwords.words('spanish'))
        fra_stops = set(stopwords.words('french'))
        
    combined_stops = ita_stops.union(spa_stops).union(fra_stops).union(ENGLISH_STOP_WORDS)
    
    extracted_keywords = [lemmatizer.lemmatize(w) for w in words if w not in combined_stops]
    return ", ".join(list(dict.fromkeys(extracted_keywords)))


def phase1_generate_keywords(
    query: str,
    llm_instant: LLM,
    pm: PromptManager,
    hint="",
    stream_placeholder=None,
):
    wordnet_keywords_str = extract_wordnet_query_keywords(query)
    wordnet_keywords = [k.strip() for k in wordnet_keywords_str.split(",") if k.strip()]

    system_prompt = pm.render(
        "keyword_generator",
        "system_prompt"
    )

    user_prompt = pm.render(
        "keyword_generator",
        "user_prompt",
        question=query,
        raw_keywords_str=wordnet_keywords_str,
        keyword_hint=hint
    )

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]

    raw_stream = ""
    tokens = 0
    print("[phase1 keyword stream] ", end="", flush=True)
    for chunk in llm_instant.stream_chat(messages):
        delta = chunk.delta or ""
        if delta:
            raw_stream += delta
            print(delta, end="", flush=True)
            if stream_placeholder is not None:
                stream_placeholder.markdown(raw_stream)

        if chunk.raw:
            prompt_tokens = chunk.raw.get("prompt_eval_count") or 0
            completion_tokens = chunk.raw.get("eval_count") or 0
            if prompt_tokens or completion_tokens:
                tokens = prompt_tokens + completion_tokens
    print("", flush=True)

    raw_content = raw_stream.strip().lower()
    extracted = list(set(
        wordnet_keywords + re.findall(r"\b[a-z0-9_]+\b", raw_content)
    ))[:15]
    return extracted, raw_content, tokens


def phase1_retrieve_candidates(
    keywords: list[str],
    solr_client: LocalSolrClient,
    all_files: list[str],
    stream_callback: StreamCallback | None = None,
) -> tuple[list[str], SolrMetadata, list[str]]:
    """Retrieve candidate files after Phase 1 keyword generation."""
    activity_log_parts: list[str] = []
    candidates: list[str] = []
    metadata: SolrMetadata = {}
    query_text = " ".join(keywords)

    try:
        print(
            "\n[phase1 candidates] Solr search "
            f"q={format_cli_log_value(query_text)} "
            f"csv_count={len(all_files)}",
            flush=True,
        )
        solr_response = solr_client.select(tokens=keywords, q_op="OR", rows=30)
        response_body = solr_response.get("response", {})
        docs = response_body.get("docs", [])
        print(
            "[phase1 candidates] Solr response "
            f"numFound={response_body.get('numFound', 'unknown')} "
            f"docs_returned={len(docs)}",
            flush=True,
        )

        for doc in docs:
            matched = match_local_csv(doc, all_files)
            if matched is None or matched in candidates:
                continue

            candidates.append(matched)
            metadata[matched] = solr_metadata_from_doc(doc)
            if len(candidates) >= 10:
                break

        if not candidates:
            candidates = all_files[:5]
            print(
                "[phase1 candidates] no local Solr matches; "
                f"fallback={candidates}",
                flush=True,
            )
        else:
            print(f"[phase1 candidates] matched={candidates}", flush=True)
    except Exception as solr_err:
        candidates = all_files[:5]
        print(
            "[phase1 candidates] Solr error "
            f"{type(solr_err).__name__}: {solr_err}; fallback={candidates}",
            flush=True,
        )

    emit_candidate_summary(candidates, metadata, activity_log_parts, stream_callback)
    return candidates, metadata, activity_log_parts
