"""Fill evaluation_section.template.tex with the generated tables and figure.

Writes paper/evaluation_section.tex from the template, results/section_tables.json
(section_tables.py) and the failure rows of results/tables.tex (latex.py).

  python3 experiments/paper_analysis/build_section.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    snippets = json.loads((HERE / "results" / "section_tables.json").read_text())
    tables = (HERE / "results" / "tables.tex").read_text()
    snippets["FAILURE_ROWS"] = tables.split("% failure rows\n", 1)[1].strip()
    text = (HERE / "evaluation_section.template.tex").read_text()
    for name, value in snippets.items():
        text = text.replace(f"%%{name}%%", value)
    left = re.findall(r"%%[A-Z0-9_]+%%", text)
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    out = ROOT / "paper" / "evaluation_section.tex"
    out.write_text(text)
    print(f"wrote {out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
