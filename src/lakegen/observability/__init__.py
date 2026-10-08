"""What a run leaves behind: its result record, its manifest, its log rows.

``recorder`` assembles the ``QueryResult`` and writes the logs from what the
pipeline's stages report; the manifest, the CSV logger and the tracing
helpers still live at their old paths until phase 4 moves them here.
"""
