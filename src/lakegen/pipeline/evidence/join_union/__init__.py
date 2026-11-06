"""The ``check_join_union`` tool's evidence: do two tables join, and do they union?

``schema_matching`` asks Valentine which columns of the two tables correspond. For a join,
``join_keys`` measures a candidate key against the values (cardinality, shared values, the size of
the merge and what a plain `pd.merge` would do), ``key_search`` ranks the candidates and looks for
composite keys, ``full_scan`` counts the best ones over every row of their columns, and
``join_report`` writes them. For a union, ``union_mapping`` maps the columns one to one and tiers
the coverage. ``column_values`` brings the values of both sides to a comparable form, ``layout``
keeps each section of the report within its room, and ``pair_check`` reads a pair of tables, runs all
of it and writes the report the agents read.
"""

from lakegen.pipeline.evidence.join_union.pair_check import check_join_union

__all__ = ["check_join_union"]
