"""Persistent HTTP bridge for Pneuma's isolated Python environment."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import inspect
import json
import os
import types

import bm25s
from bootstrap_pneuma import configure_pneuma_indexing, _validate_embedding
from pneuma import Pneuma
import Stemmer
from pneuma_judge import (
    ChatTransport,
    StructuredRelevanceJudge,
    order_by_relevance,
    structured_relevance_prompt,
)
from pneuma_oci import (
    DEFAULT_OCI_EMBED_MODEL,
    DEFAULT_OCI_LLM_MODEL,
    OCICompatOpenAI,
    OCISettings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--out-path", default="pneuma-out")
    parser.add_argument("--provider", choices=("openai", "oci"), default="openai")
    parser.add_argument(
        "--openai-base-url",
        default="http://127.0.0.1:11434/v1",
        help="OpenAI-compatible endpoint used for query embeddings and reranking",
    )
    parser.add_argument("--embedding-batch-size", default=16, type=int)
    parser.add_argument("--oci-config-file")
    parser.add_argument("--oci-profile")
    parser.add_argument("--oci-llm-model", default=DEFAULT_OCI_LLM_MODEL)
    parser.add_argument("--oci-embedding-model", default=DEFAULT_OCI_EMBED_MODEL)
    parser.add_argument("--oci-embedding-dimensions", type=int)
    parser.add_argument(
        "--judge-workers",
        default=8,
        type=int,
        help="Concurrent LLM Judge requests per query; verdicts do not depend on it",
    )
    parser.add_argument(
        "--judge-max-attempts",
        default=3,
        type=int,
        help="Attempts per document before a query fails for want of a valid verdict",
    )
    return parser


def configure_pneuma_query_tokenization(query_processor: types.ModuleType) -> None:
    """Tokenize full-text queries the way Pneuma tokenized the index.

    Pneuma 0.0.4 hands ``BM25.retrieve`` the raw query string, which bm25s
    reads as one query per character, so every full-text score is zero. It
    also scores the documents only the vector retriever found with unstemmed
    tokens, which miss stemmed index terms (``salary`` against ``salari``).
    The index was built with English stopwords and the English stemmer, so
    both query paths now tokenize that way. Clients still send the question.
    """

    def tokenize(texts, **kwargs):
        kwargs.setdefault("stopwords", "en")
        kwargs.setdefault("stemmer", Stemmer.Stemmer("english"))
        kwargs.setdefault("show_progress", False)
        return bm25s.tokenize(texts, **kwargs)

    class IndexTokenizedBM25(bm25s.BM25):
        def retrieve(self, query_tokens, *args, **kwargs):
            if isinstance(query_tokens, str):
                query_tokens = tokenize(query_tokens)
            return super().retrieve(query_tokens, *args, **kwargs)

    # The query processor reaches bm25s only through its module-level name, so
    # the substitute applies to querying; index generation keeps real bm25s.
    # ``BM25.load`` builds ``cls``, so loaded indexes get the retrieve above.
    query_bm25s = types.SimpleNamespace(**vars(bm25s))
    query_bm25s.tokenize = tokenize
    query_bm25s.BM25 = IndexTokenizedBM25
    query_processor.bm25s = query_bm25s


def judge_transport(llm: object, default_model: str) -> ChatTransport:
    """Send schema-constrained judge requests through the query processor's client."""
    if callable(getattr(llm, "chat_completion", None)):  # OCICompatOpenAI

        def oci_transport(messages, *, schema_name, schema, temperature, seed, max_tokens):
            return llm.chat_completion(
                messages,
                model=None,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                seed=seed,
                json_schema=schema,
                json_schema_name=schema_name,
            )

        return oci_transport
    if hasattr(llm, "chat"):  # an OpenAI-compatible endpoint

        def openai_transport(messages, *, schema_name, schema, temperature, seed, max_tokens):
            response = llm.chat.completions.create(
                model=default_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                seed=seed,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema, "strict": True},
                },
            )
            return response.choices[0].message.content or ""

        return openai_transport
    raise TypeError(
        f"the structured judge needs an OpenAI-compatible client, not {type(llm).__name__}"
    )


def configure_pneuma_structured_judge(
    query_processor: types.ModuleType, *, workers: int, max_attempts: int
) -> None:
    """Replace Pneuma's free-text LLM Judge with a schema-constrained one.

    Pneuma 0.0.4 samples its judge at temperature 0.7 and reads the verdict
    with ``startswith("yes")``; ``pneuma_judge`` explains how both break the
    paper's judge and how the replacement works. The prompts stay Pneuma's own,
    for content and context documents alike, with only the answer instruction
    changed, and §5.1's reordering is unchanged.
    """
    processor = query_processor.QueryProcessor
    relevance_prompt = processor._QueryProcessor__get_relevance_prompt
    # Pneuma's judge model name, which only the OpenAI-compatible path sends.
    default_model = inspect.signature(query_processor.prompt_openai_llm).parameters[
        "model"
    ].default

    def rerank(self, nodes, query):
        prompts = [
            structured_relevance_prompt(
                relevance_prompt(
                    self,
                    document,
                    "content" if node_id.split("_SEP_")[1].startswith("contents") else "context",
                    query,
                )
            )
            for node_id, _score, document in nodes
        ]
        judge = StructuredRelevanceJudge(
            judge_transport(self.pipe, default_model),
            max_attempts=max_attempts,
            workers=workers,
        )
        judgments = judge.judge_all(prompts)
        print(
            f"[judge] {sum(j.relevant for j in judgments)}/{len(judgments)} documents relevant",
            flush=True,
        )
        return order_by_relevance(nodes, judgments)

    processor._QueryProcessor__rerank = rerank


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.judge_workers < 1:
        parser.error("--judge-workers must be at least 1")
    if args.judge_max_attempts < 1:
        parser.error("--judge-max-attempts must be at least 1")
    if args.provider == "openai":
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url.rstrip("/")
        os.environ.setdefault("OPENAI_API_KEY", "ollama")

    configure_pneuma_indexing(
        embedding_batch_size=args.embedding_batch_size,
        chroma_insert_batch_size=1_000,
    )
    import pneuma.index_generator.index_generator as pneuma_index_generator
    import pneuma.query_processor.query_processor as pneuma_query_processor

    configure_pneuma_query_tokenization(pneuma_query_processor)
    configure_pneuma_structured_judge(
        pneuma_query_processor,
        workers=args.judge_workers,
        max_attempts=args.judge_max_attempts,
    )
    if args.provider == "oci":
        def prompt_oci_query_embed(embed_model, documents, model=None):
            vectors = embed_model.create_embeddings(
                list(documents), input_type="SEARCH_QUERY"
            )
            return [_validate_embedding(vector) for vector in vectors]

        pneuma_query_processor.prompt_openai_embed = prompt_oci_query_embed
    else:
        # Pneuma imports the helper into each module, so patch the query alias too.
        pneuma_query_processor.prompt_openai_embed = (
            pneuma_index_generator.prompt_openai_embed
        )

    backend = Pneuma(
        out_path=args.out_path,
        use_local_model=False,
        openai_api_key=(
            os.environ["OPENAI_API_KEY"]
            if args.provider == "openai"
            else "oci-adapter"
        ),
    )
    if args.provider == "oci":
        oci_client = OCICompatOpenAI(
            OCISettings(
                llm_model=args.oci_llm_model,
                embedding_model=args.oci_embedding_model,
                embedding_dimensions=args.oci_embedding_dimensions,
                config_file=args.oci_config_file,
                profile=args.oci_profile,
            )
        )
        backend.llm = oci_client
        backend.embed_model = oci_client

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"status": "not_found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path != "/query":
                self._send(404, {"status": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                result = backend.query_index(
                    request["index_name"],
                    request["query"],
                    k=int(request.get("k", 10)),
                    n=int(request.get("n", 5)),
                    alpha=float(request.get("alpha", 0.5)),
                )
                payload = json.loads(result) if isinstance(result, str) else result
                self._send(200, payload)
            except Exception as exc:
                self._send(
                    500,
                    {"status": "ERROR", "message": f"{type(exc).__name__}: {exc}"},
                )

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Pneuma service listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
