"""Setuptools entry point, for one reason only: the wheel tag.

``pyproject.toml`` carries the metadata, but setuptools decides
``Root-Is-Purelib`` -- and therefore ``py3-none-any`` versus a binary tag --
from ``Distribution.has_ext_modules()``, which is False for a package that only
ships ``.py`` files.  This package also ships ``libxpuint8.so``, a real ELF, so
a wheel built without this would claim to be installable on any platform and
fail to load on every one that is not Linux/x86.
"""

import os

from setuptools import Distribution, setup

try:                                            # setuptools >= 70.1
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:                             # older, via the wheel package
    from wheel.bdist_wheel import bdist_wheel


# The platform tag.
#
# PyPI does not accept the plain ``linux_x86_64`` that bdist_wheel produces.
# Its list (warehouse/utils/wheel.py) is: ``any``, ``win*``, the legacy
# ``manylinux1/2010/2014`` tags, ``linux_armv6l``/``linux_armv7l``, and the
# PEP 600/656 families ``manylinux_X_Y_arch`` and ``musllinux_X_Y_arch``.
# Anything else is refused with "400 Bad Request: Binary wheel ... has an
# unsupported platform tag 'linux_x86_64'".
#
# So the tag is a manylinux one, and it names the oldest glibc the payload is
# expected to work with.  The number is a promise in both directions: pip
# refuses the wheel on an older glibc, and the build must not reference
# symbols newer than the tag claims.  The CI job checks the second half
# (objdump on the .so) and fails the release if it does not hold.
#
# The strict manylinux promise ("glibc and nothing else") is not met: the .so
# links libsycl.so.8 and is not bundled with it (see build.sh).  That is the
# same deal as Intel's own runtime wheels on PyPI, which are manylinux-tagged
# and depend on an external runtime of their own.  The release notes say so.
PLATFORM_TAG = os.environ.get("XPUINT8_PLATFORM_TAG", "manylinux_2_35_x86_64")


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class BinaryWheel(bdist_wheel):
    """``py3-none-manylinux_2_35_x86_64``: a binary payload, but no extension.

    ``has_ext_modules`` alone would tag the wheel ``cp314-cp314-...``, which
    would need one wheel per Python version for a package that has no compiled
    module at all -- the .so is loaded through ctypes, so the ABI is torch's, not
    CPython's.
    """

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False
        self.plat_name = PLATFORM_TAG
        self.plat_name_supplied = True

    def get_tag(self):
        _impl, _abi, _plat = super().get_tag()
        return "py3", "none", PLATFORM_TAG


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": BinaryWheel})
