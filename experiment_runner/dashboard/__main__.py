"""Serve the run dashboard with uvicorn."""

from __future__ import annotations

import argparse

import uvicorn

from experiment_runner.dashboard.app import create_app


def main() -> None:
    """Parse the host and port, then serve the dashboard."""
    parser = argparse.ArgumentParser(prog="python -m experiment_runner.dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
