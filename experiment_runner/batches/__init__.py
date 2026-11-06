"""Batches of questions under one experiment configuration, run in the background.

``server`` is the batch server (LakeGen's API plus the batch endpoints),
``batch`` runs a submitted batch and writes its report, ``jobs`` keeps the
jobs on disk, ``metrics`` computes a report's table-selection and code
metrics, and ``results`` reads a stored ``QueryResult`` (its ranking,
selection, outcome and costs).
"""
