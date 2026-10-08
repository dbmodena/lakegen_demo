"""The surfaces people use: the API, the command line, the Chainlit app and the dashboard.

Only this package may import web and UI frameworks. ``api`` holds the
HTTP API (``src/api.py`` is its old entry); the CLI's body and the
Chainlit app's widgets still live at src/cli.py and lakegen.ui until
phase 4 moves them here.
"""
