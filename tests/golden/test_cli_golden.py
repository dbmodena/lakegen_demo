"""Goldens of the step-by-step CLI, ``cli.run_cli_workflow``.

The interaction golden is the terminal transcript, prompts and the scripted
answers included; it must never change. The pipeline golden is the call log
and the log rows; it changes only in listed commits.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from golden.harness import (
    COMMON_NAMES,
    Case,
    Script,
    discovery_result,
    experiment,
    make_runtime,
    mask_text,
    phase1_result,
    phase3_result,
    prepare_run,
)

SCENARIOS: dict[str, Callable[[], tuple[Case, list[str]]]] = {}


def scenario(name: str):
    def register(build):
        SCENARIOS[name] = build
        return build
    return register


SELECTED = (["gold.csv"], ["parks"], {"gold.csv": {"title": "gold"}}, "The table covers the question.", "trace", 5)
RESELECTED = (["other.csv"], ["parks", "2024"], {"other.csv": {"title": "other"}}, "other.csv has the 2024 rows.", "trace", 5)
JUDGED = (["gold.csv"], ["gold.csv", "other.csv"], {"gold.csv": {"title": "gold"}}, "The selection covers the question.", "trace", 4)
ANSWER = ("There are 42 parks.", 3)
REJECTION = phase3_result(
    rejected_reason="gold.csv has no 2024 rows.", raw_result=None, structured_result=None,
    rejection_keep_tables=[], rejection_skip_tables=["gold.csv"],
)
ERROR = phase3_result(
    error="KeyError: 'district'", raw_result=None, structured_result=None,
    execution_error={"stage": "execution", "category": "runtime_error", "retryable": True},
)
FILES = ("gold.csv", "other.csv")


def unified(**overrides):
    return experiment(interaction_mode="human_gated", **overrides)


def divided(**overrides):
    return experiment(interaction_mode="human_gated", discovery_architecture="divided", **overrides)


@scenario("unified_accept")
def _():
    return Case(unified(), Script(phase12=[SELECTED], phase3=[phase3_result()], phase4=[ANSWER]), files=FILES), ["y"]


@scenario("unified_reject_with_hint")
def _():
    script = Script(phase12=[SELECTED, RESELECTED], phase3=[phase3_result()], phase4=[ANSWER])
    return Case(unified(), script, files=FILES), ["n", "look for the 2024 edition", "y"]


@scenario("unified_orchestrated_accept")
def _():
    script = Script(unified_orchestrated=[discovery_result(["gold.csv"])], phase3=[phase3_result()], phase4=[ANSWER])
    return Case(unified(tool_access="orchestrated_context"), script, files=FILES), ["y"]


@scenario("unified_empty_retries_use_best_available")
def _():
    # Three rejected rounds: the last one forces a selection from the lake's
    # tables, which is coded without a review (the engine's policy since the
    # CLI adopted it; the CLI used to give up here).
    script = Script(
        phase12=[([], ["parks"], {}, "REJECT_KEYWORDS: nothing matched", "trace", 5)],
        phase3=[phase3_result()], phase4=[ANSWER],
    )
    return Case(unified(), script, files=FILES), []


@scenario("divided_accept")
def _():
    script = Script(phase1=[phase1_result()], phase2=[JUDGED], phase3=[phase3_result()], phase4=[ANSWER])
    return Case(divided(), script, files=FILES), ["y", "y"]


@scenario("divided_edit_keywords")
def _():
    script = Script(
        phase1=[phase1_result(), phase1_result(("parks", "district", "2024"))],
        phase2=[JUDGED], phase3=[phase3_result()], phase4=[ANSWER],
    )
    return Case(divided(), script, files=FILES), ["n", "add the year", "y", "y"]


@scenario("divided_reject_tables")
def _():
    script = Script(
        phase1=[phase1_result(), phase1_result(("parks", "2024"))],
        phase2=[JUDGED, (["other.csv"], ["other.csv"], {}, "other.csv matches the hint.", "trace", 4)],
        phase3=[phase3_result()], phase4=[ANSWER],
    )
    return Case(divided(), script, files=FILES), ["y", "n", "use the 2024 table", "y", "y"]


@scenario("divided_judge_rejects_keywords")
def _():
    script = Script(
        phase1=[phase1_result(), phase1_result(("parks", "district", "count"))],
        phase2=[([], ["gold.csv"], {}, "REJECT_KEYWORDS: the keywords only find fishing licences", "trace", 4), JUDGED],
        phase3=[phase3_result()], phase4=[ANSWER],
    )
    return Case(divided(), script, files=FILES), ["y", "y", "y"]


@scenario("coder_rejection_force")
def _():
    script = Script(phase12=[SELECTED], phase3=[REJECTION, phase3_result()], phase4=[ANSWER])
    return Case(unified(), script, files=FILES), ["y", "y"]


@scenario("coder_rejection_rediscover_accept")
def _():
    script = Script(phase12=[SELECTED, RESELECTED], phase3=[REJECTION, phase3_result()], phase4=[ANSWER])
    return Case(unified(), script, files=FILES), ["y", "n", "y"]


@scenario("coder_rejection_rediscover_abort")
def _():
    script = Script(phase12=[SELECTED, RESELECTED], phase3=[REJECTION])
    return Case(unified(), script, files=FILES), ["y", "n", "n"]


@scenario("execution_error_retry")
def _():
    script = Script(phase12=[SELECTED], phase3=[ERROR, phase3_result()], phase4=[ANSWER])
    return Case(unified(), script, files=FILES), ["y"]


@scenario("execution_error_exhausted")
def _():
    script = Script(phase12=[SELECTED], phase3=[ERROR], phase4=[("No answer could be computed.", 2)])
    return Case(unified(), script, files=FILES), ["y"]


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_cli_golden(name, golden, monkeypatch, tmp_path, capsys):
    import src.cli as cli

    case, answers = SCENARIOS[name]()
    replacements, log, capture = prepare_run(monkeypatch, tmp_path, cli, case, COMMON_NAMES)
    remaining = iter(answers)

    def scripted_input(prompt: str = "") -> str:
        try:
            answer = next(remaining)
        except StopIteration:
            raise AssertionError(f"the scenario ran out of answers at the prompt {prompt!r}") from None
        print(f"{prompt}{answer}")  # keeps the dialogue in the transcript
        return answer

    monkeypatch.setattr("builtins.input", scripted_input)

    cli.run_cli_workflow(case.question, make_runtime(case.experiment, tmp_path))

    transcript = mask_text(capsys.readouterr().out, replacements).splitlines()
    assert next(remaining, None) is None, "the scenario scripted more answers than the CLI asked for"
    golden(f"cli/{name}.transcript", {"transcript": transcript})
    golden(f"cli/{name}", {"calls": log.calls, **capture.payload()})
