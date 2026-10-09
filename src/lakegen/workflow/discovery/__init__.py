"""Table discovery: which tables of the lake answer the question.

``keywords`` writes the divided architecture's keywords, ``divided`` judges
the retrieved candidates with tools, ``unified`` searches, inspects and
selects in one agent, ``orchestrated`` runs the tool-free paths over a
prepared context; ``candidates`` holds what they share. ``tools`` holds the
unified agent's tools and ``selectors`` the table selectors a run may
configure instead of an architecture's own LLM judge.
"""
