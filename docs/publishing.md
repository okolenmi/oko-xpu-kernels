# Publishing

The short version: bump `pyproject.toml` and `python/xpu_kernels/__init__.py`,
tag `vX.Y.Z`, push the tag. `.github/workflows/release.yml` builds both
artifacts, uploads them to PyPI, and attaches the same bytes to the GitHub
Release. Nothing else is allowed to publish.

## One-time setup (already done, recorded so it can be redone)

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

On GitHub, **Settings -> Environments -> `pypi`** exists so the publish job can
be gated (required reviewers, tag restriction). GitHub creates it implicitly on
first use if it does not exist, but then it has no protection rules, which
defeats the point.

## Releasing

```bash
# 1. the version, in both places (they are compared at publish time to the tag)
$EDITOR pyproject.toml python/xpu_kernels/__init__.py

# 2. locally, before tagging -- an upload cannot be replaced
./wheel.sh --sdist && ./wheel.sh
python -m twine check dist/*
python -m pip install --no-deps --target /tmp/chk dist/*.whl
PYTHONPATH=/tmp/chk python -c 'import xpu_kernels; print(xpu_kernels.available())'

# 3. ship
git commit -am "0.2.1"
git tag -a v0.2.1 -m "0.2.1, built with oneAPI 2025.3"
git push origin main v0.2.1
```

The tag message is where the toolchain goes. The version cannot carry it: PyPI
rejects local version identifiers (`0.2.1+oneapi2025.3`) because PEP 440
reserves them for private builds, and pip ignores them when matching a
requirement anyway.

A tag that disagrees with `pyproject.toml` fails the `publish` job before
anything is uploaded. If the same version is uploaded twice, the second attempt
fails; bump the version rather than deleting the release (PyPI allows yanking,
not replacement).

## What each artifact is for

* **sdist** -- installs anywhere with no toolchain. `import xpu_kernels` works,
  `available()` is `False`, INT8 calls raise a message naming `build.sh`.
* **wheel** (`py3-none-linux_x86_64`) -- carries `libxpuint8.so` linked against
  the oneAPI 2025.x SYCL runtime. `pip` prefers it on Linux/x86-64, so that is
  what a normal `pip install xpu-kernels` gets. On a machine with a different
  runtime it degrades to the sdist behaviour; the fix is `./build.sh` from the
  unpacked sdist and `XPU_KERNELS_LIB` pointing at the result.

Both are also attached to the GitHub Release, which is the way to pin one exact
binary without an index.

## Dry run

`workflow_dispatch` is deliberately not wired up: a manual run from a branch
would exercise the OIDC claims with the wrong ref. To rehearse without
publishing, run the two build jobs' commands locally (they are the same commands
`wheel.sh` already runs) and check `twine check` output. TestPyPI would need a
second pending publisher on <https://test.pypi.org>; not worth it for a package
whose failure mode is "the upload was rejected", which is loud and free.
