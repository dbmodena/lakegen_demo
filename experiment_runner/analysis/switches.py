"""Which provider each result of a hand-switched job ran on.

While the grids of October 2026 ran, the OCI credit for gpt-oss-120b ran
out, so the model, the semantic code judge and the Pneuma entity model were
switched to Together by hand: the API was stopped, the job's stored
configuration was edited to the new value, a record was appended to the
job's ``provider_switches``, and the API was restarted to resume the job.
The stored configuration therefore holds the value a job ended with; the
results written before a switch ran with the value its record names as
``from``.

Four kinds of record exist:

* ``{at, fields, from, to, results_before_switch, reason}``: the listed
  fields switched;
* ``{at, from, to, results_before_switch, reason}``: the benchmark-lake
  grid's switch of the model and the judge, before the fields were written;
* ``{at, what: <field>, from, to}``: one field switched (the entity model);
* ``{at, what: "requeued", removed_results, reason}``: results removed and
  their questions re-queued; no value changed.

A result ran with ``to`` when its manifest timestamp is at or after the
switch. Where a record also counts the results before the switch, the count
agrees with the timestamps on every stored job (3,217 result/switch pairs,
checked on 2026-10-08). The values are written the way the user saw them at
the time (``"openai.gpt-oss-120b (OCI)"``); the model id is the part before
the parenthesis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

MODEL_FIELDS = ("model", "semantic_code_judge_model", "retrieval.pneuma_entity_model")
# The benchmark-lake grid's records without `fields` switched the model and the judge.
FIELDLESS_SWITCH = ("model", "semantic_code_judge_model")


@dataclass(frozen=True)
class ProviderSwitch:
    """One value switch of a job's ``provider_switches``.

    Attributes:
        at: When the switch was made.
        fields: The configuration fields switched (dotted keys).
        from_value: The model id the fields held before, if recorded.
        to_value: The model id the fields hold after, if recorded.
        reason: Why the switch was made, empty when not recorded.
    """

    at: datetime
    fields: tuple[str, ...]
    from_value: str | None
    to_value: str | None
    reason: str = ""


def model_id(value: Any) -> str | None:
    """``"openai.gpt-oss-120b (OCI)"`` -> ``"openai.gpt-oss-120b"``."""
    if value is None:
        return None
    return str(value).split(" (", 1)[0].strip()


def parse_timestamp(value: str) -> datetime:
    """An ISO timestamp as a datetime, reading a trailing ``Z`` as UTC."""
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def parse_switches(records: Iterable[Mapping[str, Any]]) -> list[ProviderSwitch]:
    """The value switches of a job's ``provider_switches``, oldest first; re-queues are not switches."""
    switches = []
    for record in records:
        what = record.get("what")
        if what == "requeued":
            continue
        if record.get("fields"):
            fields = tuple(record["fields"])
        elif what:
            fields = (str(what),)
        else:
            fields = FIELDLESS_SWITCH
        switches.append(
            ProviderSwitch(
                at=parse_timestamp(record["at"]),
                fields=fields,
                from_value=model_id(record.get("from")),
                to_value=model_id(record.get("to")),
                reason=str(record.get("reason") or ""),
            )
        )
    return sorted(switches, key=lambda switch: switch.at)


def values_at(
    stored: Mapping[str, Any],
    switches: Iterable[ProviderSwitch],
    timestamp: str | None,
) -> dict[str, str | None]:
    """The model fields as they were for a result started at ``timestamp``.

    ``stored`` is the recorded configuration's flat values (dotted keys),
    which hold what the job ended with; each later switch restores its
    ``from`` value, latest first, so a result before two switches of one
    field gets the earliest value. Without a timestamp the stored values
    are returned.

    Args:
        stored: The recorded configuration's flat values.
        switches: The job's switches (``parse_switches``).
        timestamp: When the result started (ISO), or None.

    Returns:
        The value of each of ``MODEL_FIELDS`` when the result started.
    """
    values: dict[str, str | None] = {key: stored.get(key) for key in MODEL_FIELDS}
    if timestamp is None:
        return values
    started = parse_timestamp(timestamp)
    for switch in sorted(switches, key=lambda switch: switch.at, reverse=True):
        if started < switch.at:
            for key in switch.fields:
                values[key] = switch.from_value
    return values
