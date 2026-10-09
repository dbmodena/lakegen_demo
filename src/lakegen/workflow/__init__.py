"""The question's workflow: its phases, the pipeline that runs them, and what records the run.

``discovery`` finds the tables (phases 1 and 2), ``coding`` writes and runs
the analysis code (phase 3), ``evaluation`` judges what the code returned,
``answer`` writes the reply (phase 4). ``pipeline`` is the engine that runs
them in order and handles the gates; ``observability`` records the run;
``prompts`` holds the templates the phases send.
"""
