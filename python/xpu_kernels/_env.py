"""Triton's device-extension default, set before the package imports any Triton.

Triton's Intel backend asks an OpenCL platform for the card's extensions; a Mesa rusticl platform answers for the same card and
does not list the matrix units, so ``tl.dot`` lowers to FMA and the attention crawls (docs/experiments.md 5c).  These are the two
extensions every gate in this package stays open with.  The bf16-conversion extension is faster for the attention but closes the
rope's bit-exact gate, so it stays opt-in.  A value already in the environment always wins.
"""

from __future__ import annotations

import os

DEFAULT_DEVICE_EXTENSIONS = "cl_intel_subgroup_matrix_multiply_accumulate cl_intel_subgroup_2d_block_io"
os.environ.setdefault("TRITON_INTEL_DEVICE_EXTENSIONS", DEFAULT_DEVICE_EXTENSIONS)
