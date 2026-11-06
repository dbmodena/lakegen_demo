"""The command line: the ``lakegen`` command and the step-by-step terminal driver.

``main`` routes ``lakegen <command>`` to the command's own parser; ``ask``
is the question command (``src/cli.py`` is its old entry); ``terminal``
holds the prompts and the colours the CLI has always used; ``driver``
runs the pipeline step by step with them.
"""
