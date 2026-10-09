from types import SimpleNamespace

import pytest

from lakegen.judge_json import chat_json_with_repair, extract_json


def test_extract_json_parses_a_bare_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_a_markdown_fence():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_scans_past_leading_and_trailing_prose():
    assert extract_json('Here is the result: {"a": 1} -- done.') == {"a": 1}


def test_extract_json_raises_on_non_object():
    with pytest.raises(Exception):
        extract_json("[1, 2, 3]")


def test_extract_json_raises_on_no_json_at_all():
    with pytest.raises(Exception):
        extract_json("not json at all")


class _FakeLlm:
    def __init__(self, contents):
        self._contents = list(contents)

    def chat(self, _messages):
        content = self._contents.pop(0)
        return SimpleNamespace(message=SimpleNamespace(content=content, additional_kwargs={}), raw={})


def test_chat_json_with_repair_parses_a_clean_first_response():
    result = chat_json_with_repair(_FakeLlm(['{"ok": true}']), "prompt", repair_fields="ok")
    assert result == {"ok": True}


def test_chat_json_with_repair_falls_back_to_a_second_call():
    result = chat_json_with_repair(
        _FakeLlm(["not json", '{"ok": true}']), "prompt", repair_fields="ok",
    )
    assert result == {"ok": True}


def test_chat_json_with_repair_raises_when_both_attempts_fail():
    with pytest.raises(Exception):
        chat_json_with_repair(_FakeLlm(["nope", "still nope"]), "prompt", repair_fields="ok")
