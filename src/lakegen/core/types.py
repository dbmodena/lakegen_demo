from typing import Any, Callable

SolrMetadata = dict[str, dict[str, Any]]
Phase2SelectionResult = tuple[list[str], list[str], SolrMetadata, str, str, int]
StreamCallback = Callable[[str], None]
