#!/usr/bin/env python3
"""Write docs/configuration.md from the environment registry.

    python scripts/gen_config_docs.py

tests/config/test_settings.py fails when the file is out of date.
"""

from pathlib import Path

from lakegen.config.settings import render_markdown

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "configuration.md"


def main() -> None:
    """Write docs/configuration.md and print its path."""
    TARGET.parent.mkdir(exist_ok=True)
    TARGET.write_text(render_markdown(), encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
