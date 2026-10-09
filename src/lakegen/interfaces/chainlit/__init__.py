"""The Chainlit app's driver of the engine; the only package that imports chainlit.

``driver`` runs the pipeline's steps on a worker thread, shows each stage
as a Chainlit step, answers each gate with the app's buttons and prompts.
The app's entry (``src/app.py``) and its widgets stay in ``lakegen.ui``
until phase 4 moves them here.
"""
