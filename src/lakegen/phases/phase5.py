"""Retired compatibility shim.

The Chainlit app now treats synthesis as Phase 4. This module is intentionally
not exported from lakegen.phases anymore, but the wrapper keeps old direct
imports from crashing while external callers migrate.
"""

from lakegen.phases.phase4 import phase4_synthesize


def phase5_synthesize(query, raw_result, llm, pm):
    return phase4_synthesize(query, raw_result, llm, pm)
