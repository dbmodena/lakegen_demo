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

    _ERROR_PATTERNS = [
        "error:",
        "error ",
        "exception:",
        "exception ",
        "traceback",
        "errno",
        "no such file",
        "filenotfounderror",
        "permissionerror",
        "modulenotfounderror",
        "importerror",
        "keyerror",
        "valueerror",
        "typeerror",
        "indexerror",
        "zerodivisionerror",
    ]

    try:
        result = subprocess.run([sys.executable, str(fp)],
                                capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            stdout_lower = result.stdout.lower()
            if any(pat in stdout_lower for pat in _ERROR_PATTERNS):
                return None, result.stdout.strip(), code
            return result.stdout.strip(), None, code

        # Non-zero return code — build a detailed error message
        detail = result.stderr.strip() or result.stdout.strip()
        error_msg = f"[Exit code {result.returncode}] {detail}"
        return None, error_msg, code
    except subprocess.TimeoutExpired:
        return None, "Execution timed out (15s limit).", code
    except Exception as e:
        return None, f"[{type(e).__name__}] {e}", code
