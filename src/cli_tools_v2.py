"""Run the LakeGen CLI with the isolated ``agent_tools_2`` implementation.

This launcher installs the v2 Phase-2 and Phase-1/2 tool modules before the
workflow is imported.  The regular ``src/cli.py`` remains the baseline.
"""

from __future__ import annotations

import sys

from lakegen.agent_tools_2 import tools_p12 as tools_p12_v2
from lakegen.agent_tools_2 import tools_p2 as tools_p2_v2


sys.modules["lakegen.agent_tools.tools_p2"] = tools_p2_v2
sys.modules["lakegen.agent_tools.tools_p12"] = tools_p12_v2

from src.cli import main  # noqa: E402 - aliases must be registered first


if __name__ == "__main__":
    main()
