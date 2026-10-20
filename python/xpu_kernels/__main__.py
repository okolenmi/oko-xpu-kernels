"""``python -m xpu_kernels`` — self-test what is installed on this machine.

Checks the INT8 library end to end and then reports the state of the Triton
kernels.  The Triton self-tests need a reference implementation to compare
against (see ``xpu_kernels.triton.<module>.set_reference``); without one they
report ``unverified`` and stay disabled, which is the intended behaviour — a
kernel whose numerics nobody has pinned down is not worth the speed.
"""

from __future__ import annotations

import sys

from . import int8, triton


def main() -> int:
    ok = int8.selftest()
    print()
    for status in triton.status():
        print(f"{status}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
