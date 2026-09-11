"""Persistent HTTP bridge for Pneuma's isolated Python environment."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json

from pneuma import Pneuma


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--out-path", default="pneuma-out")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = Pneuma(
        out_path=args.out_path,
        use_local_model=True,
        llm_path=args.llm_model,
        embed_path=args.embedding_model,
    )

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
