"""The entry points: the API, the command line and the web app.

Only this package may import web and UI frameworks. ``api`` holds the HTTP
API (``python -m lakegen.entrypoints.api``), ``cli`` the ``lakegen`` command,
``webapp`` the Chainlit app (``chainlit run src/lakegen/entrypoints/webapp/app.py``).
"""
