from unittest.mock import patch

import requests

from scout.data import http


def _response(status, body="ok"):
    response = requests.Response()
    response.status_code = status
    response._content = body.encode()
    return response


def test_caches_200_but_never_403(tmp_path):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=30):
        calls.append(url)
        return _response(403 if "blocked" in url else 200)

    with patch.object(http.requests, "get", fake_get), patch.object(http.time, "sleep"):
        http.polite_get("https://h.test/a", cache_dir=tmp_path)
        http.polite_get("https://h.test/a", cache_dir=tmp_path)
        http.polite_get("https://h.test/blocked", cache_dir=tmp_path)
        http.polite_get("https://h.test/blocked", cache_dir=tmp_path)
    assert calls == ["https://h.test/a", "https://h.test/blocked", "https://h.test/blocked"]


def test_spaces_live_requests_per_host(tmp_path):
    slept = []
    clock = iter([100.0, 100.0, 100.5, 100.5])
    with (
        patch.object(http.requests, "get", lambda *a, **k: _response(200)),
        patch.object(http.time, "sleep", slept.append),
        patch.object(http.time, "monotonic", lambda: next(clock)),
    ):
        http.polite_get("https://h.test/1", cache_dir=tmp_path)
        http.polite_get("https://h.test/2", cache_dir=tmp_path)
    assert slept and abs(slept[-1] - 2.5) < 1e-6


def test_params_are_part_of_the_cache_key(tmp_path):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=30):
        calls.append(params)
        return _response(200)

    with patch.object(http.requests, "get", fake_get), patch.object(http.time, "sleep"):
        http.polite_get("https://h.test/x", params={"offset": 0}, cache_dir=tmp_path)
        http.polite_get("https://h.test/x", params={"offset": 100}, cache_dir=tmp_path)
        http.polite_get("https://h.test/x", params={"offset": 0}, cache_dir=tmp_path)
    assert calls == [{"offset": 0}, {"offset": 100}]


def test_empty_body_is_not_cached(tmp_path):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=30):
        calls.append(url)
        return _response(200, body="" if len(calls) == 1 else "{}")

    with patch.object(http.requests, "get", fake_get), patch.object(http.time, "sleep"):
        first = http.polite_get("https://h.test/soft", cache_dir=tmp_path)
        second = http.polite_get("https://h.test/soft", cache_dir=tmp_path)
    assert first.text == "" and second.text == "{}" and len(calls) == 2


def test_transport_failure_status_raises_and_is_not_cached(tmp_path):
    import pytest

    calls = []

    def fake_get(url, params=None, headers=None, timeout=30):
        calls.append(url)
        return _response(
            0 if len(calls) == 1 else 200, body="failed to do request" if len(calls) == 1 else "{}"
        )

    with patch.object(http.requests, "get", fake_get), patch.object(http.time, "sleep"):
        with pytest.raises(ConnectionError):
            http.polite_get("https://h.test/t", cache_dir=tmp_path)
        assert http.polite_get("https://h.test/t", cache_dir=tmp_path).text == "{}"
    assert len(calls) == 2
