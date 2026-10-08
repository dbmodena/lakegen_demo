"""A dashboard over the run store: every experiment, its cells, their questions.

    python -m lakegen.interfaces.dashboard [--host 127.0.0.1] [--port 8792]

It reads what ``lakegen.analysis`` reads (the store's index, then the jobs
it points at), so it shows any experiment the store knows: the specs run
with ``exp run`` and the legacy grids indexed with ``index-legacy``. With
``LAKEGEN_DASHBOARD_TOKEN`` set, open ``/?token=...`` once; the token is
then kept in a cookie.
"""

from lakegen.interfaces.dashboard.app import create_app

__all__ = ["create_app"]
