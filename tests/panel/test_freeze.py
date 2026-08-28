import pandas as pd
import pytest

from scout.panel import freeze, player_match

AS_OF = pd.Timestamp("2021-07-01")


def test_seasons_are_known_once_they_have_ended():
    assert freeze.seasons_before([2019, 2020, 2021], AS_OF) == [2019, 2020]
    with pytest.raises(freeze.LeakageError, match=r"\[2021\]"):
        freeze.refuse_after([2020, 2021], AS_OF)


def test_cut_and_assert_by_date_and_by_season():
    dated = pd.DataFrame({"date": ["2021-05-01", "2021-08-14"], "x": [1, 2]})
    assert freeze.cut(dated, AS_OF, date_col="date").x.tolist() == [1]
    with pytest.raises(freeze.LeakageError, match="1 rows"):
        freeze.assert_frozen(dated, AS_OF, date_col="date")
    by_season = pd.DataFrame({"season": [2020, 2021]})
    assert freeze.cut(by_season, AS_OF).season.tolist() == [2020]


def test_player_match_refuses_a_post_freeze_season(monkeypatch):
    rows = pd.DataFrame(
        {
            "league": ["L"] * 2,
            "season": [2020, 2021],
            "team_id": [1, 1],
            "player_id": [7, 7],
            "position": ["MC", "MC"],
            "minutes": [90, 90],
            "date": ["2021-05-01", "2021-08-14"],
        }
    )
    monkeypatch.setattr(player_match.understat, "load", lambda *args, **kwargs: rows)
    with pytest.raises(freeze.LeakageError):
        player_match.build(seasons=[2020, 2021], as_of=AS_OF)
    trimmed = player_match.build(as_of=AS_OF)  # no explicit season asked for: later rows are cut
    assert trimmed.season.tolist() == [2020]
