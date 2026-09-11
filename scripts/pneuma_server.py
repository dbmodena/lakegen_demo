"""Persistent HTTP bridge for Pneuma's isolated Python environment."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os

from bootstrap_pneuma import configure_pneuma_indexing, _validate_embedding
from pneuma import Pneuma
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.provider == "openai":
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url.rstrip("/")
        os.environ.setdefault("OPENAI_API_KEY", "ollama")

    configure_pneuma_indexing(
        embedding_batch_size=args.embedding_batch_size,
        chroma_insert_batch_size=1_000,
    )
    import pneuma.index_generator.index_generator as pneuma_index_generator
    import pneuma.query_processor.query_processor as pneuma_query_processor

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
