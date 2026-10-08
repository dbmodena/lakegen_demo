"""Reading what experiments produced: stored jobs, their results and the metrics.

``results`` reads one stored ``QueryResult`` (the ranking, the selection, the
outcome, the costs); ``metrics`` computes a batch's table-selection report
from its questions and results, the way the API logs it. ``jobs`` reads an
API job directory, ``switches`` says which provider each result ran on when a
job's models were switched by hand, and ``runs`` turns experiments or job
directories into one table of runs for analysis.
"""
