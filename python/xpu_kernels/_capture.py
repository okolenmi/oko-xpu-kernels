"""Is a device-graph capture in progress?

The one question every capture-safety check in this package has to ask, kept
in its own module so that it can be imported from anywhere without a cycle:
``int8/ops.py`` needs it, and so does the Triton gate.
"""

from __future__ import annotations

import torch

__all__ = ["is_capturing"]


def is_capturing() -> bool:
    """True while a device graph is being captured on the current stream.

    Safe to call anywhere and never raises: it answers ``False`` when there is
    no accelerator, and a backend that cannot tell us is treated as "not
    capturing" -- the conservative answer for a check that exists to *refuse*
    something would be the opposite, so the callers of this function treat
    "unknown" as "not capturing" and rely on the other checks instead.

    A capture is a property of the *stream*, not of the process: the same
    process can capture one region and run eagerly outside it, which is why
    this cannot be a flag the caller sets.
    """
    for name in ("xpu", "cuda"):
        backend = getattr(torch, name, None)
        if backend is None:
            continue
        probe = getattr(backend, "is_current_stream_capturing", None)
        if probe is None:
            continue
        try:
            if probe():
                return True
        except Exception:  # noqa: BLE001 - a backend that cannot answer is not capturing
            continue
    return False
