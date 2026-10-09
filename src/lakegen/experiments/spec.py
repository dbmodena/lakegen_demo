"""An experiment spec and the cells it expands to.

A spec replaces a submit script::

    name: qwen-grid
    description: Full pipeline on local Qwen 27B, 5 lakes x 6 modes x 2 selectors
    base: [config/base.yaml, config/presets/model/ollama_qwen3_27b.yaml]
    set: {discovery.architecture: divided}
    grid:
      lake: [valencia, paris, bologna, nyc, uk]
      retrieval.mode: [keyword, semantic, hybrid, duckdb_agentic, pneuma, pneuma_seeker]
      discovery.selector.kind: [llm, jev]
    per_mode:                       # settings a grid value brings with it
      retrieval.mode:
        pneuma_seeker: {retrieval.seeker.content_fusion: fill, retrieval.seeker.enumerate_tables: false}
    per_lake:
      uk: {retrieval.pneuma.base_url: http://localhost:8767}
    questions: {set: "100q_{lake}", passes: [{name: a, slice: ":20"}, {name: b, slice: "20:"}]}
    experiment_id: "{name}-{discovery.selector.kind}-{pass}-{lake}-{retrieval.mode}"
    execution: {backend: api, api_url: http://127.0.0.1:8006, workers: 2, stagger_seconds: 240}
    requires: ["pneuma:{lake}", ollama]

``base`` files and presets compose in order (the nested layout of
``lakegen.config.files``), ``set`` applies on top, then the grid point, then
the per-value settings. Keys may use either layout's spelling. A cell is one
grid point and one question slice; its id is a hash of the resolved
configuration and the questions, so re-running a spec finds its cells again.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from lakegen.config.experiment import ExperimentConfig, load_experiment_config
from lakegen.config.files import deep_merge, load_document, nest, translate_override
from lakegen.config.locations import BASE_DIR
from lakegen.config.registry import get_lake
from lakegen.config.retrieval import RetrievalMode


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QuestionPass(StrictModel):
    name: str
    slice: str | None = None  # "start:stop" in Python slice notation


class QuestionSelection(StrictModel):
    """Which benchmark questions a cell runs.

    ``set`` names a file under benchmark/ without the extension and may use
    ``{lake}`` or any grid axis; ``file`` gives a path instead. ``slice``
    narrows the cases; ``passes`` splits them into named slices that become
    separate cells.
    """

    set: str | None = None
    file: str | None = None
    slice: str | None = None
    passes: list[QuestionPass] = Field(default_factory=list)


class ModelSplit(StrictModel):
    model: str
    share: float = Field(ge=0.0, le=1.0)


class Execution(StrictModel):
    backend: str = "api"  # api | local
    api_url: str = "http://127.0.0.1:8002"
    job_dir: str | None = None
    workers: int = Field(default=1, ge=1)
    stagger_seconds: float = Field(default=0.0, ge=0.0)
    split: ModelSplit | None = None


class ExperimentSpec(StrictModel):
    name: str
    description: str = ""
    base: list[str] = Field(default_factory=list)
    set: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    # Explicit grid points for runs the cartesian product cannot express; a
    # point may carry its own `questions`. They add to the grid's points.
    cells: list[dict[str, Any]] = Field(default_factory=list)
    per_mode: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)
    per_lake: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # The Pneuma service of a lake for this run (base URL), where it differs
    # from config/lakes.yaml; applied to the Pneuma cells only, as a
    # LAKEGEN_PNEUMA_<LAKE>_BASE_URL in the submitter's environment was.
    pneuma_routes: dict[str, str] = Field(default_factory=dict)
    questions: QuestionSelection = Field(default_factory=QuestionSelection)
    experiment_id: str = "{name}-{axes}"
    execution: Execution = Field(default_factory=Execution)
    requires: list[str] = Field(default_factory=list)
    # Fields the run did not hold fixed (dotted keys): switched by hand while
    # it ran, so the spec's value is the intended one, not what every job
    # used. They are not compared when stored jobs are matched to cells;
    # `lakegen.analysis.load_runs` reports what each result ran with.
    varied: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSpec:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(document)


@dataclass(frozen=True)
class Cell:
    """One grid point with one question slice, fully resolved."""

    spec_name: str
    axes: dict[str, Any]          # grid key (flat spelling) -> value
    pass_name: str | None
    questions_file: Path
    questions_slice: str | None
    config: ExperimentConfig

    @property
    def lake(self) -> str:
        return self.config.core

    @property
    def experiment_id(self) -> str:
        return self.config.experiment_id

    @property
    def cell_id(self) -> str:
        payload = {
            "config": self.config.model_dump(mode="json"),
            "questions": str(self.questions_file),
            "slice": self.questions_slice,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        axes = " ".join(f"{key}={value}" for key, value in self.axes.items())
        suffix = f" pass={self.pass_name}" if self.pass_name else ""
        return f"{self.experiment_id:40} {axes}{suffix}  {self.questions_file.name}[{self.questions_slice or ':'}]"


def _flat(overrides: Mapping[str, Any]) -> dict[str, Any]:
    return nest({translate_override(key): value for key, value in overrides.items()})


def _dotted_get(data: Mapping[str, Any], dotted: str) -> Any:
    cursor: Any = data
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _template_values(point: Mapping[str, Any], document: Mapping[str, Any]) -> dict[str, Any]:
    """The grid point in both spellings, plus the document's ``lake`` and ``mode``."""
    values: dict[str, Any] = {}
    for key, value in point.items():
        values[key] = value
        values[translate_override(key)] = value
    values["lake"] = document.get("core")
    values["mode"] = _dotted_get(document, "retrieval.mode")
    return values


def _format(template: str, values: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"{key!r} in {template!r} is not a grid axis, 'name', 'pass' or 'axes'")
        return str(values[key])

    return re.sub(r"\{([^{}]+)\}", replace, template)


def _slices(selection: QuestionSelection) -> list[tuple[str | None, str | None]]:
    if selection.passes:
        return [(item.name, item.slice) for item in selection.passes]
    return [(None, selection.slice)]


def compose(spec: ExperimentSpec, point: Mapping[str, Any]) -> dict[str, Any]:
    """The flat configuration document of one grid point, before validation."""
    data: dict[str, Any] = {}
    for base in spec.base:
        data = deep_merge(data, load_document(BASE_DIR / base))
    data = deep_merge(data, _flat(spec.set))
    data = deep_merge(data, _flat(point))
    for axis, by_value in spec.per_mode.items():
        value = _dotted_get(data, translate_override(axis))
        if value is not None and str(value) in by_value:
            data = deep_merge(data, _flat(by_value[str(value)]))
    lake, mode = data.get("core"), _dotted_get(data, "retrieval.mode")
    if lake and mode and RetrievalMode(mode).is_pneuma:
        # Pin the lake's Pneuma service as the API would resolve it at run time
        # (the registry's route with the submitter's overrides), so the cell's
        # configuration says which service it ran against.
        try:
            url, index = get_lake(lake).pneuma_route()
        except KeyError:
            pass  # an unknown lake is reported by the validation below
        else:
            url = spec.pneuma_routes.get(lake, url)
            data = deep_merge(data, {"retrieval": {"pneuma_base_url": url, "pneuma_index_name": index}})
    if lake in spec.per_lake:
        data = deep_merge(data, _flat(spec.per_lake[lake]))
    return data


def expand(spec: ExperimentSpec) -> list[Cell]:
    """Every cell of the spec: the grid's cartesian product times the question slices."""
    cells: list[Cell] = []
    for point, selection in _points(spec):
        document = compose(spec, point)
        template_values = _template_values(point, document)
        for pass_name, questions_slice in _slices(selection):
            values = {**template_values, "name": spec.name, "pass": pass_name or "",
                      "axes": "-".join(str(value) for value in point.values())}
            experiment_id = _format(spec.experiment_id, values).strip("-")
            questions_file = _questions_file(selection, values)
            document_with_id = deep_merge(document, {
                "experiment_id": experiment_id,
                "benchmark": {"path": str(questions_file)},
            })
            config = load_experiment_config(data_override=document_with_id)
            cells.append(Cell(
                spec_name=spec.name,
                axes={translate_override(key): value for key, value in point.items()},
                pass_name=pass_name,
                questions_file=questions_file,
                questions_slice=questions_slice,
                config=config,
            ))
    return cells


def _points(spec: ExperimentSpec) -> Iterator[tuple[dict[str, Any], QuestionSelection]]:
    """The grid's points, then the explicit cells, each with its question selection."""
    if spec.grid:
        keys = list(spec.grid)
        for values in itertools.product(*(spec.grid[key] for key in keys)):
            yield dict(zip(keys, values, strict=True)), spec.questions
    elif not spec.cells:
        yield {}, spec.questions
    for point in spec.cells:
        point = dict(point)
        selection = point.pop("questions", None)
        yield point, QuestionSelection.model_validate(selection) if selection else spec.questions


def _questions_file(selection: QuestionSelection, values: Mapping[str, Any]) -> Path:
    if selection.file:
        return BASE_DIR / _format(selection.file, values)
    if selection.set:
        return BASE_DIR / "benchmark" / f"{_format(selection.set, values)}.json"
    raise ValueError("questions needs a 'set' or a 'file'")
