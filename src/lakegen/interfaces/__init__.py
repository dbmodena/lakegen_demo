"""The surfaces people use: the API, the command line, the Chainlit app and the dashboard.

Only this package may import web and UI frameworks. ``api`` holds the
HTTP API, ``cli`` the ``lakegen`` command, ``chainlit`` the app's driver and
widgets, ``dashboard`` the retrieval dashboard; ``src/api.py``, ``src/cli.py``
and ``src/app.py`` are the process entry points.
"""
