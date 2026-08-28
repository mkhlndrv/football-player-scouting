from unittest.mock import patch

import pandas as pd
import requests

from scout.data import clubelo

CSV = (
    "Rank,Club,Country,Level,Elo,From,To\n"
    "5,Arsenal,ENG,1,1900.5,2023-08-01,2023-08-13\n"
    "4,Arsenal,ENG,1,1910.0,2023-08-14,2023-08-20\n"
)
EMPTY = "Rank,Club,Country,Level,Elo,From,To\n"


def _response(body):
    response = requests.Response()
    response.status_code = 200
    response._content = body.encode()
    return response


def test_fetch_club_parses_dates_and_uses_spaceless_key():
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        return _response(CSV)

    with patch.object(clubelo, "polite_get", fake_get):
        history = clubelo.fetch_club("Man City")
    assert seen == ["http://api.clubelo.com/ManCity"]
    assert history.From.dtype.kind == "M" and len(history) == 2 and history.Elo.iloc[1] == 1910.0


def test_unknown_club_returns_empty_frame():
    with patch.object(clubelo, "polite_get", lambda url, **kwargs: _response(EMPTY)):
        assert clubelo.fetch_club("Flamengo").empty


def test_elo_on_picks_interval_and_nan_outside():
    with patch.object(clubelo, "polite_get", lambda url, **kwargs: _response(CSV)):
        history = clubelo.fetch_club("Arsenal")
    assert clubelo.elo_on(history, pd.Timestamp("2023-08-15")) == 1910.0
    assert clubelo.elo_on(history, pd.Timestamp("2023-08-05")) == 1900.5
    assert pd.isna(clubelo.elo_on(history, pd.Timestamp("2024-01-01")))
    assert pd.isna(clubelo.elo_on(history.iloc[0:0], pd.Timestamp("2023-08-15")))
