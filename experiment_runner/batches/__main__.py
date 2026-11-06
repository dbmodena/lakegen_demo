"""Serve the batch server with uvicorn."""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    """Parse the host and port, then serve the batch server."""
    parser = argparse.ArgumentParser(prog="python -m experiment_runner.batches")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    uvicorn.run("experiment_runner.batches.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
