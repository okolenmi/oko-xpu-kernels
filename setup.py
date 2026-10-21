"""Setuptools entry point, for one reason only: the wheel tag.

``pyproject.toml`` carries the metadata, but setuptools decides
``Root-Is-Purelib`` -- and therefore ``py3-none-any`` versus
``py3-none-linux_x86_64`` -- from ``Distribution.has_ext_modules()``, which is
False for a package that only ships ``.py`` files.  This package also ships
``libxpuint8.so``, a real ELF, so a wheel built without this would claim to be
installable on any platform and fail to load on every one that is not Linux/x86.
It also links ``libsycl.so.8`` rather than bundling it (see ``build.sh``), so
``linux_x86_64`` is the honest tag; it is not manylinux.
"""

from setuptools import Distribution, setup

try:                                            # setuptools >= 70.1
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:                             # older, via the wheel package
    from wheel.bdist_wheel import bdist_wheel


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class BinaryWheel(bdist_wheel):
    """``py3-none-linux_x86_64``: a binary payload, but no Python extension.

    ``has_ext_modules`` alone would tag the wheel ``cp314-cp314-...``, which
    would need one wheel per Python version for a package that has no compiled
    module at all -- the .so is loaded through ctypes, so the ABI is torch's, not
    CPython's.
    """

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _impl, _abi, plat = super().get_tag()
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": BinaryWheel})
