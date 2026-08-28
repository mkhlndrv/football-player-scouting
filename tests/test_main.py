import pytest

from scout.__main__ import SOURCES, fetch


def test_fetch_only_runs_the_named_sources(monkeypatch):
    ran = []
    for name in SOURCES:
        monkeypatch.setitem(SOURCES, name, lambda name=name: ran.append(name))
    fetch(["injuries", "clubelo"])
    assert ran == ["injuries", "clubelo"]


def test_fetch_runs_everything_in_order(monkeypatch):
    ran = []
    for name in SOURCES:
        monkeypatch.setitem(SOURCES, name, lambda name=name: ran.append(name))
    fetch()
    assert ran == list(SOURCES)


def test_unknown_source_is_rejected(monkeypatch, capsys):
    import sys

    from scout.__main__ import main

    monkeypatch.setattr(sys, "argv", ["scout", "fetch", "--only", "nope"])
    with pytest.raises(SystemExit):
        main()
    assert "unknown sources" in capsys.readouterr().err
