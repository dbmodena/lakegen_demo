"""Table discovery: which tables of the lake answer the question.

The architectures themselves (the unified agent, the divided keyword writer
and judge, the orchestrated selectors) still live in ``lakegen.phases``
until phase 4.7 moves them here; ``selectors`` already holds the protocol
and the registry of the table selectors they can run instead of their own
LLM judge.
"""
