"""A dashboard over the run store: every experiment, its cells, their questions.

    python -m experiment_runner.dashboard [--host 127.0.0.1] [--port 8792]

It reads what ``experiment_runner.analysis`` reads (the store's index, then the jobs
it points at), so it shows any experiment the store knows: the specs run
with ``python -m experiment_runner run`` and the legacy grids indexed with
``index-legacy``. With
``LAKEGEN_DASHBOARD_TOKEN`` set, open ``/?token=...`` once; the token is
then kept in a cookie.
"""

from experiment_runner.dashboard.app import create_app

__all__ = ["create_app"]
