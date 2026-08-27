import json
from unittest.mock import patch

import pandas as pd
import requests

from scout.data import sofascore


def _response(payload):
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(payload).encode()
    return response


def test_list_seasons():
    payload = {"seasons": [{"id": 52186, "year": "23/24"}, {"id": 41886, "year": "22/23"}]}
    with patch.object(sofascore, "polite_get", lambda *a, **k: _response(payload)):
        seasons = sofascore.list_seasons(17)
    assert seasons.season_id.tolist() == [52186, 41886]
    assert seasons.year.tolist() == ["23/24", "22/23"]


def test_position_group_pages_and_keeps_none_as_nan():
    pages = {
        0: {
            "results": [
                {
                    "player": {"id": 1, "name": "A One"},
                    "team": {"id": 9, "name": "Arsenal"},
                    "goals": 3,
                    "tackles": 1,
                }
            ],
            "page": 1,
            "pages": 2,
        },
        100: {
            "results": [
                {
                    "player": {"id": 2, "name": "B Two"},
                    "team": {"id": 9, "name": "Arsenal"},
                    "goals": 0,
                    "tackles": None,
                }
            ],
            "page": 2,
            "pages": 2,
        },
    }

    def fake_get(url, params=None, headers=None, tls=False):
        assert params["filters"] == "position.in.F" and tls
        return _response(pages[params["offset"]])

    with patch.object(sofascore, "polite_get", fake_get):
        frame = sofascore.position_group_stats(17, 52186, "F", fields=["goals", "tackles"])
    assert frame.sofascore_player_id.tolist() == [1, 2]
    assert frame.position_group.tolist() == ["F", "F"]
    assert frame.goals.tolist() == [3, 0]
    assert frame.tackles.isna().tolist() == [False, True]


def test_pull_league_writes_one_file_per_season_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(sofascore, "RAW", tmp_path)
    monkeypatch.setattr("scout.data.retry.time.sleep", lambda s: None)
    seasons = pd.DataFrame({"season_id": [52186, 41886], "year": ["23/24", "22/23"]})
    calls = []

    def fake_group(ut_id, season_id, group, fields=None):
        calls.append((season_id, group))
        return pd.DataFrame(
            {"sofascore_player_id": [season_id * 10 + len(calls)], "position_group": [group]}
        )

    with (
        patch.object(sofascore, "list_seasons", lambda ut_id: seasons),
        patch.object(sofascore, "position_group_stats", fake_group),
    ):
        sofascore.pull_league("GB1", [2022, 2023], log=lambda m: None)
        first_round = len(calls)
        sofascore.pull_league("GB1", [2022, 2023], log=lambda m: None)
    assert first_round == 8 and len(calls) == 8  # 4 groups x 2 seasons, nothing re-pulled
    saved = sofascore.load()
    assert set(saved.season) == {2022, 2023} and set(saved.competition_id) == {"GB1"}
    assert len(saved) == 8


def test_season_year_for_calendar_leagues():
    assert sofascore.season_year("BRA1", 2023) == "2023"
    assert sofascore.season_year("GB1", 2023) == "23/24"
