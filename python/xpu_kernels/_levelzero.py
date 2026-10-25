"""Find the Level Zero headers Triton needs, so no one has to export a path before every command.

Triton's Intel backend compiles a small launcher (C++) for every new kernel argument list and includes ``level_zero/ze_api.h``;
it looks for it through ``LEVEL_ZERO_V1_SDK_PATH`` (``<path>/include`` and ``<path>/lib``) and the compiler's default include
directories.  Distributions that do not ship the headers where Triton looks (CachyOS, any install without root) fail at the first
kernel with a page of unrelated compiler warnings.  This sets the variable when it is unset and a directory with the headers
is found in one of the usual places; an explicit setting always wins.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

HEADER = Path("include") / "level_zero" / "ze_api.h"
VARIABLE = "LEVEL_ZERO_V1_SDK_PATH"


def candidates() -> list[Path]:
    home = Path.home()
    return [home / ".local" / "levelzero", home / ".local", Path("/opt/intel/level-zero"), Path("/opt/level-zero")]


def ensure_headers() -> str | None:
    """The SDK path in effect after this call, or None when none was found (the system includes may still work)."""
    current = os.environ.get(VARIABLE)
    if current:
        return current
    for root in candidates():
        if (root / HEADER).is_file():
            os.environ[VARIABLE] = str(root)
            log.info("%s set to %s", VARIABLE, root)
            return str(root)
    return None
