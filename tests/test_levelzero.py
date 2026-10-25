"""``_levelzero.ensure_headers``: an explicit setting wins, a directory with the headers is found, nothing found changes nothing."""

import os

from xpu_kernels import _levelzero as L


def _sdk(root):
    header = root / L.HEADER
    header.parent.mkdir(parents=True)
    header.write_text("")
    return root


def test_explicit_setting_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(L.VARIABLE, "/somewhere")
    monkeypatch.setattr(L, "candidates", lambda: [_sdk(tmp_path / "sdk")])
    assert L.ensure_headers() == "/somewhere"
    assert os.environ[L.VARIABLE] == "/somewhere"


def test_a_directory_with_the_headers_is_found(monkeypatch, tmp_path):
    monkeypatch.delenv(L.VARIABLE, raising=False)
    monkeypatch.setattr(L, "candidates", lambda: [tmp_path / "empty", _sdk(tmp_path / "sdk")])
    assert L.ensure_headers() == str(tmp_path / "sdk")
    assert os.environ[L.VARIABLE] == str(tmp_path / "sdk")


def test_nothing_found_changes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv(L.VARIABLE, raising=False)
    monkeypatch.setattr(L, "candidates", lambda: [tmp_path / "empty"])
    assert L.ensure_headers() is None
    assert L.VARIABLE not in os.environ
