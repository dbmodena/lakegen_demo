import re
import subprocess
import sys
import uuid
from pathlib import Path

from src.utils import BASE_DIR


def phase4_execute(code_raw, run_dir: Path | None = None):
    match = re.search(r"```python\n(.*?)\n```", code_raw, re.DOTALL)
    code = match.group(1).strip() if match else code_raw.replace("```python","").replace("```","").strip()

    forbidden = ["import os","import sys","import shutil","subprocess","eval(","exec("]
    if any(f in code for f in forbidden):
        return None, "Security Error: Forbidden libraries used.", code

    coding_dir = run_dir or BASE_DIR / "coding" / uuid.uuid4().hex
    coding_dir.mkdir(parents=True, exist_ok=True)
    fp = coding_dir / "script.py"
    fp.write_text(code, encoding="utf-8")

    try:
        result = subprocess.run([sys.executable, str(fp)],
                                capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            stdout_lower = result.stdout.lower()
            if "error:" in stdout_lower or "exception:" in stdout_lower:
                return None, result.stdout.strip(), code
            return result.stdout.strip(), None, code
        return None, (result.stderr.strip() or result.stdout.strip()), code
    except Exception as e:
        return None, str(e), code
