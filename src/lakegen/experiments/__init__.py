"""Experiments as data: a spec names the grid, the questions and how to run them.

``spec`` turns a YAML spec into cells (one resolved configuration and one
question slice each), ``questions`` reads the benchmark files, ``store``
keeps what was submitted under the state root, and ``executors`` submit
cells to a LakeGen API or run them in process. ``python -m
lakegen.experiments.cli`` is the command line until the ``lakegen`` command
exists.
"""
