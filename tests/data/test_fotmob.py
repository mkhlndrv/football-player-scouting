import json
from unittest.mock import patch

import requests

from scout.data import fotmob

SEASONS = [
    {"id": 20957, "name": "2023/2024", "leagueId": 40},
    {"id": 12768, "name": "2018/2019", "leagueId": 40},
]
DEEP_EMPTY_SEASON = {
    "statsData": [],
    "seasons": SEASONS,
    "statsList": [{"name": "goals", "title": "Top scorer"}],
}
DEEP = {
    "statsData": [],
    "seasons": SEASONS,
    "statsList": [
        {"name": "mins_played", "title": "Minutes played", "category": "Top Stat"},
        {"name": "total_tackle", "title": "Tackles per 90", "category": "Defending"},
    ],
}
STATIC = {
    "mins_played": {
        "TopLists": [
            {
                "StatList": [
                    {
                        "ParticipantName": "A One",
                        "ParticiantId": 1,
                        "TeamId": 9,
                        "TeamName": "Brugge",
                        "StatValue": 3000.0,
                        "SubStatValue": 34.0,
                        "MinutesPlayed": 3000,
                        "MatchesPlayed": 34,
                        "Rank": 1,
                        "Positions": [36],
                    },
                    {
                        "ParticipantName": "B Two",
                        "ParticiantId": 2,
                        "TeamId": 9,
                        "TeamName": "Brugge",
                        "StatValue": 200.0,
                        "SubStatValue": 5.0,
                        "MinutesPlayed": 200,
                        "MatchesPlayed": 5,
                        "Rank": 2,
                        "Positions": [11],
                    },
                ]
            }
        ]
    },
    "total_tackle": {
        "TopLists": [
            {
                "StatList": [
                    {
                        "ParticipantName": "A One",
                        "ParticiantId": 1,
                        "TeamId": 9,
                        "TeamName": "Brugge",
                        "StatValue": 2.1,
                        "SubStatValue": 70.0,
                        "MinutesPlayed": 3000,
                        "MatchesPlayed": 34,
                        "Rank": 1,
                        "Positions": [36],
                    },
                ]
            }
        ]
    },
}


def _response(payload, status=200):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode()
    return response


def fake_get(url, params=None, headers=None, **kwargs):
    if "leagueseasondeepstats" in url:
        return _response(DEEP if params["season"] else DEEP_EMPTY_SEASON)
    stat = url.rsplit("/", 1)[-1].removesuffix(".json")
    return _response(STATIC[stat])


def test_seasons_and_stats_from_deep_stats_endpoint():
    with patch.object(fotmob, "polite_get", fake_get):
        seasons, stats = fotmob.league_seasons_and_stats(40)
    assert seasons == {"2023/2024": 20957, "2018/2019": 12768}
    assert stats.name.tolist() == ["mins_played", "total_tackle"]


def test_stat_table_is_long_and_raw():
    with patch.object(fotmob, "polite_get", fake_get):
        table = fotmob.stat_table(40, 20957, "total_tackle")
    assert table.stat.tolist() == ["total_tackle"]
    assert table.fotmob_player_id.tolist() == [1] and table.stat_value.tolist() == [2.1]
    assert table.minutes_played.tolist() == [3000]


def test_pull_league_writes_long_file_per_season_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(fotmob, "RAW", tmp_path)
    monkeypatch.setattr("scout.data.retry.time.sleep", lambda s: None)
    calls = []

    def counting_get(url, params=None, headers=None, **kwargs):
        calls.append(url)
        return fake_get(url, params, headers)

    with patch.object(fotmob, "polite_get", counting_get):
        fotmob.pull_league("BE1", [2023, 2018], log=lambda m: None)
        first_round = len(calls)
        fotmob.pull_league("BE1", [2023, 2018], log=lambda m: None)
    assert len(calls) == first_round  # files exist: nothing re-fetched, not even the season list
    frame = fotmob.load()
    assert set(frame.season) == {2023, 2018} and set(frame.competition_id) == {"BE1"}
    assert len(frame[(frame.season == 2023) & (frame.stat == "mins_played")]) == 2
    assert (
        len(frame[(frame.season == 2023) & (frame.stat == "total_tackle")]) == 1
    )  # absence kept raw


def test_season_name_for_calendar_leagues():
    assert fotmob.season_name("BRA1", 2023) == "2023"
    assert fotmob.season_name("BE1", 2023) == "2023/2024"


def test_stat_table_treats_404_as_empty():
    def not_found(url, params=None, headers=None, **kwargs):
        return _response({}, status=404)

    with patch.object(fotmob, "polite_get", not_found):
        table = fotmob.stat_table(40, 20957, "_goals_prevented")
    assert table.empty and "stat" in table.columns
