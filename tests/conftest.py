"""Test-suite configuration.

``needs_testing_grounds`` holds tests written against code that exists only on
the ``testing-grounds`` branch (modules such as ``lakegen.code_judge`` or the
parallel-inspection and PRP-rerank features). They are kept so that work can
be ported later, but they are not collected here.
"""

collect_ignore = ["needs_testing_grounds"]
