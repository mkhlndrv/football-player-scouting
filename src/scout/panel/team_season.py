import pandas as pd

from scout.data import understat

SIDE_STATS = ["xg", "np_xg", "ppda", "deep_completions", "expected_points", "goals"]
# notebook 01 Step 5c: every season mean has year-to-year r >= 0.58; xg duplicates np_xg
# (r 0.99), expected_points_against mirrors _for (r -1.00), goals are outcomes not style
STYLE = [
    "np_xg_for",
    "np_xg_against",
    "ppda_for",
    "ppda_against",
    "deep_completions_for",
    "deep_completions_against",
    "expected_points_for",
]
TEAM_KEYS = ["league", "season", "team_id"]


def _side(frame: pd.DataFrame, own: str, other: str) -> pd.DataFrame:
    rows = frame[["league", "season", "game_id", "date", f"{own}_team_id", f"{own}_team"]]
    rows = rows.rename(columns={f"{own}_team_id": "team_id", f"{own}_team": "team"})
    rows = rows.assign(is_home=own == "home", opponent_id=frame[f"{other}_team_id"])
    for stat in SIDE_STATS:
        rows[f"{stat}_for"] = frame[f"{own}_{stat}"]
        rows[f"{stat}_against"] = frame[f"{other}_{stat}"]
    return rows


def team_match_long(team_match: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, with the team's own and conceded numbers."""
    long = pd.concat(
        [_side(team_match, "home", "away"), _side(team_match, "away", "home")], ignore_index=True
    )
    long = long.sort_values(TEAM_KEYS + ["date"]).reset_index(drop=True)
    long["match_no"] = long.groupby(TEAM_KEYS).cumcount()
    return long


def season_means(long: pd.DataFrame) -> pd.DataFrame:
    values = [c for c in long.columns if c.endswith(("_for", "_against"))]
    means = long.groupby(TEAM_KEYS + ["team"])[values].mean()
    means["matches"] = long.groupby(TEAM_KEYS + ["team"]).size()
    return means.reset_index()


def build(leagues: list[str] | None = None, seasons: list[int] | None = None) -> pd.DataFrame:
    return season_means(team_match_long(understat.load("team_match", leagues, seasons)))
