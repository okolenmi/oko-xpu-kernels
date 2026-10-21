# Publishing

The short version: bump `pyproject.toml` and `python/xpu_kernels/__init__.py`,
tag `vX.Y.Z`, push the tag. `.github/workflows/release.yml` builds both
artifacts, uploads them to PyPI, and attaches the same bytes to the GitHub
Release. Nothing else is allowed to publish.

## One-time setup

PyPI uses **trusted publishing**: no API token is stored anywhere. The upload
job asks GitHub for an OIDC token and PyPI checks four strings. At
<https://pypi.org/manage/account/publishing/> the pending publisher is:

| Field | Value |
| --- | --- |
| PyPI Project Name | `xpu-kernels` |
| Owner | `okolenmi` |
| Repository name | `oko-xpu-kernels` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Renaming the workflow file or the repository breaks the upload with a
`invalid-publisher` / "trusted publisher was not found" error, which names none
of the fields. Keep `release.yml` and the `pypi` environment in sync with the
table.

### The `pypi` environment is optional

GitHub creates an environment the first time a job names one, with no
protection rules. So the "Environment name" field above can stay filled in and
the environment does not have to exist beforehand -- the upload works either
way. Creating it deliberately only buys protection rules (required reviewers,
a tag restriction), and the menu is easy to miss: **repository** Settings (the
gear on the repository page, not on your user page) -> left sidebar, under
**Code and automation**, next to Actions and Pages, at
<https://github.com/okolenmi/oko-xpu-kernels/settings/environments>. The
sidebar scrolls separately from the page; on a narrow window it collapses
behind the menu icon next to "Settings".

## Where the wheel is built, and why it matters

The wheel carries `libxpuint8.so`, and that `.so` links `libsycl.so.8` plus
`libsvml.so`, `libirng.so`, `libimf.so` and `libintlc.so.5` -- the SYCL
runtime of the oneAPI release it was compiled against. A consumer with a
different oneAPI *patch* release may not have the symbols, so the CI job pins
the compiler to **2025.3.2** (the release this repository was verified with)
with an apt preference, and asserts the version before building. If the pin
cannot be satisfied the job fails; it never publishes a `.so` that might not
load.

The alternative -- a self-hosted runner on the maintainer's desktop -- was
rejected: there is no runner registered, a runner daemon would sit on the
machine that does the timing measurements, and the job would queue forever.

Two smaller things the runner pins, both learned the hard way:

* **`ubuntu-24.04`, not `ubuntu-latest`.** The wheel carries an ELF, and a
  wheel built against a newer glibc does not load on an older one.
  `ubuntu-latest` migrates to Ubuntu 26.04 (glibc 2.4x) as of 2026-10-19, and
  the machines that have oneAPI 2025.3 are not all on the newest libc.
* **setuptools in the interpreter.** The wheel is built with
  `--no-build-isolation`, so the build backend has to be importable where pip
  runs. A 3.12 venv from the runner image has none (ensurepip stopped
  shipping it), and the first run of this workflow died with
  `Cannot import 'setuptools.build_meta'`. `wheel.sh` now says so in one line
  instead of letting pip say it in a stack trace.

The `device` job in `ci.yml` is a different thing and stays as it is: it needs
a real GPU, so it is skipped unless the repository variable `XPU_RUNNER` is
set to `yes`.

## Releasing

Rehearse first. The workflow has a `workflow_dispatch` trigger that builds
both artifacts and publishes nothing (the publish jobs are gated on
`github.ref_type == 'tag'`):

```bash
# Actions -> release -> Run workflow (on main)
```

That is where a broken oneAPI install shows up, without touching PyPI.

Then, for a real release:

```bash
# 1. the version, in both places (compared against the tag at publish time)
$EDITOR pyproject.toml python/xpu_kernels/__init__.py

# 2. locally, before tagging -- an upload cannot be replaced
./wheel.sh --sdist && ./wheel.sh
python -m twine check dist/*
python -m pip install --no-deps --target /tmp/chk dist/*.whl
PYTHONPATH=/tmp/chk python -c 'import xpu_kernels; print(xpu_kernels.available())'

# 3. ship
git commit -am "0.2.1"
git tag -a v0.2.1 -m "0.2.1, built with oneAPI 2025.3.2"
git push origin main v0.2.1
```

The tag message is where the toolchain goes. The version cannot carry it: PyPI
rejects local version identifiers (`0.2.1+oneapi2025.3`) because PEP 440
reserves them for private builds, and pip ignores them when matching a
requirement anyway.

A tag that disagrees with `pyproject.toml` fails the `publish` job before
anything is uploaded. If the same version is uploaded twice, the second attempt
fails; `skip-existing` is set, so a re-run of a *partially* failed release
uploads only what is missing.

## After the upload: check the published wheel

The CI build proves the wheel is well formed, not that it computes the same
numbers as the local build. The published wheel is what a consumer gets, so
install it and run the project's own gate against it:

```bash
pip install --no-deps --force-reinstall xpu-kernels==0.2.0
python -c 'import xpu_kernels; print(xpu_kernels.available(), xpu_kernels.device_name())'
# then the PNG gate, which must still produce the recorded md5
```

If the published wheel disagrees, the local build is the reference and the
release is wrong -- rebuild locally and upload that file by hand with a
project-scoped token (`twine upload dist/*.whl`); PyPI accepts another file
under the same version.

## What each artifact is for

* **sdist** -- installs anywhere with no toolchain. `import xpu_kernels` works,
  `available()` is `False`, INT8 calls raise a message naming `build.sh`.
* **wheel** (`py3-none-linux_x86_64`) -- carries `libxpuint8.so` linked against
  the oneAPI 2025.3 SYCL runtime. `pip` prefers it on Linux/x86-64, so that is
  what a normal `pip install xpu-kernels` gets. On a machine with a different
  runtime it degrades to the sdist behaviour; the fix is `./build.sh` from the
  unpacked sdist and `XPU_KERNELS_LIB` pointing at the result.

Both are also attached to the GitHub Release, which is the way to pin one exact
binary without an index.

## Dry run

`workflow_dispatch` is deliberately not wired to publish: a manual run from a
branch would exercise the OIDC claims with the wrong ref. The rehearsal above
builds and checks everything and stops there. TestPyPI would need a second
pending publisher on <https://test.pypi.org>; not worth it for a package whose
failure mode is "the upload was rejected", which is loud and free.
