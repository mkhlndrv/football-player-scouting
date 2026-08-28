import pandas as pd

# notebook 02 Step 6: on-target xG faced minus goals conceded, per 90 of the keeper's minutes.
# Year-to-year r 0.34 on 889 keeper-seasons (0.42 on the 245 with all three measures); FotMob
# goals prevented 0.26-0.36 and Sofascore saves per 90 (volume, r -0.20 with the proxy) lose.
ON_TARGET = ("Goal", "Saved Shot")
MATCH_KEYS = ["competition_id", "season", "game_id"]


def on_target_faced(shots: pd.DataFrame) -> pd.DataFrame:
    """Per (match, shooting team): xG of on-target shots and goals, own goals excluded."""
    on_target = shots[shots["result"].isin(ON_TARGET)]
    grouped = on_target.groupby(MATCH_KEYS + ["team_id"])
    return grouped.agg(
        xg_on_target=("xg", "sum"), goals=("result", lambda r: int((r == "Goal").sum()))
    ).reset_index()


def keeper_matches(player_match: pd.DataFrame, min_minutes: int = 45) -> pd.DataFrame:
    keepers = player_match[
        (player_match["role"] == "GK") & (player_match["minutes"] >= min_minutes)
    ]
    return keepers[MATCH_KEYS + ["team_id", "player_id", "minutes"]]


def prevented_per90(
    player_match: pd.DataFrame, shots: pd.DataFrame, team_game: pd.DataFrame
) -> pd.DataFrame:
    """Season totals per keeper: minutes, on-target xG faced, goals conceded, prevented per 90.
    `team_game` lists both teams of every match (competition_id, season, game_id, team_id)."""
    sides = team_game[MATCH_KEYS + ["team_id"]]
    opponents = sides.merge(sides.rename(columns={"team_id": "opp_id"}), on=MATCH_KEYS)
    opponents = opponents[opponents["team_id"] != opponents["opp_id"]]
    faced = on_target_faced(shots).rename(columns={"team_id": "opp_id"})
    rows = keeper_matches(player_match).merge(opponents, on=MATCH_KEYS + ["team_id"])
    rows = rows.merge(faced, on=MATCH_KEYS + ["opp_id"], how="left").fillna(
        {"xg_on_target": 0.0, "goals": 0}
    )
    season = rows.groupby(["competition_id", "season", "player_id"]).agg(
        minutes=("minutes", "sum"),
        matches=("game_id", "size"),
        xg_on_target=("xg_on_target", "sum"),
        goals_conceded=("goals", "sum"),
    )
    season["prevented_per90"] = (
        (season["xg_on_target"] - season["goals_conceded"]) / season["minutes"] * 90
    )
    return season.reset_index()
