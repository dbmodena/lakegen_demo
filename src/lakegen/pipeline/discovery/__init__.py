"""Table discovery: which tables of the lake answer the question.

``keywords`` writes the divided architecture's keywords, ``divided`` runs
retrieval and the table selector over the retrieved candidates, ``unified``
runs the one agent that searches and selects; ``state`` is the round's record
both write. ``selectors`` holds the table selector of the divided
architecture, the MemGPT agent, which the unified agent extends with a search
tool.
"""
