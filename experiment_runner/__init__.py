"""The experiment harness: grids of LakeGen runs, the services they need, and their analysis.

LakeGen answers questions; this package runs experiments on it. A spec
(``experiments/specs``) names a grid of configurations and question slices,
``spec`` turns it into cells, ``executors`` submit them to the batch server
(``batches``: LakeGen's API plus batches run in the background) or run them in
process, ``store`` remembers what was submitted, ``services``
starts what the cells need, ``analysis`` reads the results back as one table
of runs and ``dashboard`` shows them. ``benchmark`` scores retrieval alone.
``legacy`` indexes the jobs of the old submit scripts; its configuration
comparison also checks that the specs reproduce those jobs.

Run from the repository root with LakeGen's environment:
``.venv/bin/python -m experiment_runner --help``. LakeGen never imports this
package (an import contract keeps it so).
"""
