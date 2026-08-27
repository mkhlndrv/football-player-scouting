from unittest.mock import patch

import pytest

from scout.data import retry


def test_retries_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("dropped")
        return "done"

    with patch.object(retry.time, "sleep"):
        assert retry.until_done(flaky, attempts=5, wait_s=0, log=lambda m: None) == "done"
    assert calls["n"] == 3


def test_gives_up_after_attempts():
    def always():
        raise ConnectionError("dropped")

    with patch.object(retry.time, "sleep"), pytest.raises(ConnectionError):
        retry.until_done(always, attempts=2, wait_s=0, log=lambda m: None)
