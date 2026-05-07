import os
import sys
from pathlib import Path

import nltk


def ensure_project_paths(src_dir: Path, root_dir: Path) -> None:
    for path in (src_dir, root_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def nltk_download_dir() -> Path:
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return Path(venv) / "nltk_data"

    prefix = Path(sys.prefix)
    base_prefix = Path(getattr(sys, "base_prefix", sys.prefix))
    if prefix != base_prefix:
        return prefix / "nltk_data"

    return Path.home() / "nltk_data"


def _nltk_resource_exists(resource_path: str) -> bool:
    for candidate in (resource_path, f"{resource_path}.zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def bootstrap_nltk_data() -> str | None:
    data_dir = nltk_download_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = str(data_dir)

    if data_path not in nltk.data.path:
        nltk.data.path.insert(0, data_path)

    env_paths = [p for p in os.environ.get("NLTK_DATA", "").split(os.pathsep) if p]
    if data_path not in env_paths:
        os.environ["NLTK_DATA"] = os.pathsep.join([data_path, *env_paths])

    required_packages = {
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "stopwords": "corpora/stopwords",
    }

    missing = []
    for package, resource_path in required_packages.items():
        if _nltk_resource_exists(resource_path):
            continue
        if not nltk.download(package, download_dir=data_path, quiet=True):
            missing.append(package)

    still_missing = [
        package
        for package, resource_path in required_packages.items()
        if not _nltk_resource_exists(resource_path)
    ]
    missing = sorted(set(missing + still_missing))
    if missing:
        return (
            "Missing NLTK resource(s): "
            f"{', '.join(missing)}. Download target: {data_path}"
        )
    return None


def load_css(path: Path) -> str:
    return path.read_text(encoding="utf-8")
