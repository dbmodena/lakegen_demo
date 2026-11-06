"""Serve the API with uvicorn."""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    """Parse the host and port, then serve the API."""
    parser = argparse.ArgumentParser(prog="python -m lakegen.entrypoints.api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("lakegen.entrypoints.api.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
