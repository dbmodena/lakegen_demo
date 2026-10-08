"""Where LakeGen's shared resources used to live; each now comes from its package.

Kept for the experiment scripts and notebooks that import them here: the
LLM clients are ``lakegen.llm``, the Solr client and the retrieval service
``lakegen.retrieval.factory``, the retrieval log ``lakegen.retrieval.run_log``,
the prompts ``lakegen.prompts``, the lake's tables ``lakegen.lakes``.
"""

from lakegen.lakes import get_all_csv_files, get_all_table_files  # noqa: F401
from lakegen.llm.factory import get_llm  # noqa: F401
from lakegen.llm.providers.oci import oci_runtime_config as _oci_runtime_config  # noqa: F401
from lakegen.prompts import get_prompt_manager  # noqa: F401
from lakegen.retrieval.factory import get_solr, get_table_retrieval_service  # noqa: F401
from lakegen.retrieval.run_log import (  # noqa: F401
    capture_retrieval_runs,
    get_retrieval_run_logger,
    log_retrieval_decision,
    make_retrieval_run_observer,
)
from lakegen.retrieval.run_log import (
    log_and_capture_retrieval_run as _log_and_capture_retrieval_run,  # noqa: F401
)
