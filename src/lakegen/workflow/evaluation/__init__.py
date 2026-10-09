"""Judging what the coder produced: the checks that decide whether an answer counts.

``code`` compares a code run's result with the question's reference and
labels the attempt; ``semantic_judge`` is the LLM judge that reads the
result against the question; ``output_validation`` checks the shape of a
result before either runs; ``reference_execution`` re-runs a reference
solution in a worker process (tests and benchmark builders only).
"""
