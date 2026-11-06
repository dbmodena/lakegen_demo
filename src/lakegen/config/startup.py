"""Process start-up: the ``.env`` file, ``sys.path`` and the NLTK data the keyword tools need.

Importing the module loads ``.env`` into the environment.
"""

import os
import sys
from pathlib import Path

import nltk
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def nltk_download_dir() -> Path:
    """Where NLTK data goes: the active virtualenv's ``nltk_data``, else the home folder's."""
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
    """Make wordnet, omw-1.4 and stopwords available, downloading the missing ones.

    Puts the download folder first on NLTK's path and on ``NLTK_DATA``, so
    subprocesses find the data too.

    Returns:
        A message naming the resources still missing, or None when all are there.
    """
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
        return f"Missing NLTK resource(s): {', '.join(missing)}. Download target: {data_path}"
    return None
