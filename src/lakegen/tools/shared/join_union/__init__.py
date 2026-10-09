"""The ``check_join_union`` tool's evidence: do two tables join, and do they union?

``schema_matching`` asks Valentine which columns of the two tables correspond; ``join_keys``
measures each candidate key against the values (cardinality, shared values, join size);
``union_mapping`` maps the columns one to one and tiers the coverage; ``column_values`` brings the
values of both sides to a comparable form; ``pair_check`` reads a pair of tables, runs all of it and
writes the report the agents read.
"""

from lakegen.tools.shared.join_union.pair_check import check_join_union

__all__ = ["check_join_union"]
