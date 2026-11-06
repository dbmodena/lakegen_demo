import time
from types import SimpleNamespace

import pandas as pd
from openai import APIConnectionError

from pneuma_engine.narration import cap_long_cells, concurrent_prompt_openai_llm


def test_concurrent_narration_answers_every_conversation_in_place_and_in_order():
    """Pneuma reads the answer appended to each conversation; concurrency must not reorder."""
    failures = {"c2": 1}  # one transient failure, retried like Pneuma's own loop

    def create(**request):
        column = request["messages"][0]["content"]
        time.sleep(0.01 * (5 - int(column[1])))  # later columns answer first
        if failures.get(column):
            failures[column] -= 1
            raise APIConnectionError(request=None)
        assert request["seed"] == 42 and request["max_tokens"] == 400
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"about {column}"))]
        )

    llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    conversations = [[{"role": "user", "content": f"c{index}"}] for index in range(5)]

    returned = concurrent_prompt_openai_llm(4)(llm, conversations, max_new_tokens=400)

    assert returned is conversations
    assert [conversation[-1]["content"] for conversation in conversations] == [
        f"about c{index}" for index in range(5)
    ]


def test_row_sample_cells_are_capped_without_touching_other_values():
    frame = pd.DataFrame(
        {
            "the_geom": ["MULTIPOLYGON (((" + "1 2, " * 500 + ")))", None],
            "boro": ["Brooklyn", "Queens"],
            "count": [3, 4],
        }
    )

    capped = cap_long_cells(frame, 20)

    assert capped.loc[0, "the_geom"] == frame.loc[0, "the_geom"][:20] + "…"
    assert pd.isna(capped.loc[1, "the_geom"])
    assert capped["boro"].tolist() == ["Brooklyn", "Queens"]
    assert capped["count"].tolist() == [3, 4]
    assert frame.loc[0, "the_geom"].endswith(")))")  # the input is not modified
