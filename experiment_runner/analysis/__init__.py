"""Reading what experiments produced: stored jobs and runs as one table.

``jobs`` reads an API job directory, ``switches`` says which provider each
result ran on when a job's models were switched by hand, ``runs`` turns
experiments or job directories into one table of runs, ``summary`` aggregates
it, and ``sources`` reads results that did not come from API jobs. One stored
result and a batch's metrics are read by ``experiment_runner.batches``
(``results`` and ``metrics``).
"""
