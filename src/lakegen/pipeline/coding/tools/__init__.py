"""The coder's tools: inspect the tables, run the code in the sandbox, check the result, finish or reject.

``CoderTools`` (``manager``) composes them from one mixin per
concern (``plan_grounding``, ``contract``, ``sandbox``, ``error_repair``,
``inspection``, ``rejection``, ``result_checks``, ``finish``) over the run's
``CoderState`` (``state``). The tool names and descriptions are what the
prompts name, so they do not change here.
"""

from lakegen.pipeline.coding.tools.execution.error_repair import classify_execution_error
from lakegen.pipeline.coding.tools.manager import CoderTools, RejectTablesSchema
from lakegen.pipeline.coding.tools.state import CoderLifecycle, CoderState, ReviewStatus

__all__ = [
    "CoderLifecycle",
    "CoderState",
    "CoderTools",
    "RejectTablesSchema",
    "ReviewStatus",
    "classify_execution_error",
]
