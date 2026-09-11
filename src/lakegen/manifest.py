"""Immutable, secret-safe experiment manifests."""

from __future__ import annotations

import re
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import ConfigDict, BaseModel

from lakegen.experiment_config import ExperimentConfig
from lakegen.reproducibility import initialize_reproducibility


_SECRET_KEY = re.compile(
    r"(?:secret|password|passwd|token|api[_-]?key|credential|authorization|cookie)",
    re.IGNORECASE,
)


def redact_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str) and "://" in value:
        try:
            parsed = urlsplit(value)
            hostname = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port is not None else ""
            netloc = hostname + port
            if parsed.username is not None or parsed.password is not None:
                netloc = "[REDACTED]@" + netloc
            query = urlencode([
                (key, "[REDACTED]" if _SECRET_KEY.search(key) else item)
                for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            ])
            return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
        except ValueError:
            return "[REDACTED]"
    return value


def git_version(base_dir: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class ExperimentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str
    run_id: str
    question_id: str
    timestamp: datetime
    git_version: str
    model: str
    core: str
    dataset: str
    seed: int
    representation_version: str
    retrieval_parameters: dict[str, Any]
    resolved_config: dict[str, Any]
    reproducibility: dict[str, Any]


def create_manifest(
    config: ExperimentConfig,
    *,
    base_dir: Path,
    question: str,
    question_id: str | int | None = None,
    run_id: str | None = None,
) -> ExperimentManifest:
    resolved = redact_secrets(config.model_dump(mode="json"))
    retrieval = redact_secrets(config.retrieval.model_dump(mode="json"))
    stable_question_id = str(question_id) if question_id is not None else uuid.uuid5(
        uuid.NAMESPACE_URL, question.strip()
    ).hex
    return ExperimentManifest(
        experiment_id=config.experiment_id,
        run_id=run_id or uuid.uuid4().hex,
        question_id=stable_question_id,
        timestamp=datetime.now(timezone.utc),
        git_version=git_version(base_dir),
        model=config.model,
        core=config.core,
        dataset=config.core,
        seed=config.seed,
        representation_version=config.retrieval.representation_version,
        retrieval_parameters=retrieval,
        resolved_config=resolved,
        reproducibility=initialize_reproducibility(config.seed).telemetry(),
    )


def persist_manifest(manifest: ExperimentManifest, directory: Path) -> Path:
    """Create a manifest once; an existing run manifest is never overwritten."""

    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{manifest.run_id}.json"
    with target.open("x", encoding="utf-8") as output:
        json.dump(manifest.model_dump(mode="json"), output, ensure_ascii=False, indent=2)
        output.write("\n")
    target.chmod(0o444)
    return target
