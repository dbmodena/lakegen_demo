"""Compatibility alias: the prompt manager lives in ``lakegen.prompts``.

Both names refer to one module object. Remove once nothing imports
``prompts.prompt_manager``.
"""

import sys

from lakegen.prompts import prompt_manager as _prompt_manager

sys.modules[__name__] = _prompt_manager
