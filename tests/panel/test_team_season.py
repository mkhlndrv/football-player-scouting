import pandas as pd

from scout.panel.team_season import SIDE_STATS, STYLE, season_means, team_match_long


def _matches():
    base = {
        "league": ["L", "L"],
        "season": [2023, 2023],
        "game_id": [1, 2],
        "date": ["2023-08-12", "2023-08-19"],
        "home_team_id": [10, 20],
        "away_team_id": [20, 10],
        "home_team": ["A", "B"],
        "away_team": ["B", "A"],
    }
    for stat in SIDE_STATS:
        base[f"home_{stat}"] = [2.0, 1.0]
        base[f"away_{stat}"] = [0.5, 3.0]
    return pd.DataFrame(base)


def test_long_form_swaps_for_and_against_per_side():
    long = team_match_long(_matches())
    assert len(long) == 4
    team_a = long[long.team_id == 10].sort_values("date")
    assert team_a.np_xg_for.tolist() == [2.0, 3.0]
    assert team_a.np_xg_against.tolist() == [0.5, 1.0]
    assert team_a.is_home.tolist() == [True, False]
    assert team_a.match_no.tolist() == [0, 1]


def test_season_means_average_all_matches_and_expose_style_columns():
    means = season_means(team_match_long(_matches())).set_index("team_id")
    assert means.loc[10, "np_xg_for"] == 2.5 and means.loc[20, "np_xg_for"] == 0.75
    assert means.loc[10, "matches"] == 2
    assert set(STYLE) <= set(means.columns)
