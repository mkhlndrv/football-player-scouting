import pandas as pd

# notebook 02 Step 1: 600 minutes in a role-season is where year-to-year stability stops rising
# materially while 77.6% of the backtest departures are kept
MIN_MINUTES = 600
PENALTY_XG = 0.7612  # Understat's fixed penalty xG (notebook 01, Part 4b)
ROLE_KEYS = ["competition_id", "season", "player_id", "role"]

UNDERSTAT = ["npxg", "xa", "key_passes", "shots", "xg_chain", "xg_buildup"]
WORKRATE = [
    "tackles",
    "interceptions",
    "recoveries",
    "clearances",
    "dribbles",
    "key_passes",
    "big_chances_created",
    "accurate_passes",
    "accurate_long_balls",
    "fouls",
    "possession_won_att_third",
]
# quantities with year-to-year r >= 0.3 at MIN_MINUTES per role; goals and assists are never
# inputs (spec 4.A) and were the least stable columns in every role
ROLE_QUANTITIES = {
    "GK": {
        "understat": ["xg_chain", "xg_buildup"],
        "workrate": ["recoveries", "clearances", "accurate_passes", "accurate_long_balls", "saves"],
    },
    "CB": {
        "understat": [q for q in UNDERSTAT if q != "xa"],
        "workrate": [m for m in WORKRATE if m != "big_chances_created"],
    },
    "FB": {"understat": UNDERSTAT, "workrate": WORKRATE},
    "CM": {"understat": UNDERSTAT, "workrate": WORKRATE},
    "W": {"understat": UNDERSTAT, "workrate": WORKRATE},
    "ST": {"understat": UNDERSTAT, "workrate": WORKRATE},
}
SUMMED = ["xg", "xa", "key_passes", "shots", "xg_chain", "xg_buildup", "goals", "assists"]


def penalty_xg(shots: pd.DataFrame) -> pd.Series:
    """Penalty xG per (game, player): Understat files penalties as situation NA at 0.7612."""
    is_penalty = shots["situation"].isna() & (shots["xg"].round(4) == PENALTY_XG)
    return shots[is_penalty].groupby(["game_id", "player_id"])["xg"].sum().rename("pen_xg")


def season_role_totals(player_match: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Minutes and summed quantities per player x league-season x role, npxG from the shots."""
    rows = player_match.dropna(subset=["role"]).merge(
        penalty_xg(shots), on=["game_id", "player_id"], how="left"
    )
    rows["pen_xg"] = rows["pen_xg"].fillna(0.0)
    rows["npxg"] = rows["xg"] - rows["pen_xg"]
    sums = {q: (q, "sum") for q in ["npxg", *SUMMED]}
    return rows.groupby(ROLE_KEYS).agg(minutes=("minutes", "sum"), **sums).reset_index()


def season_role_per90(player_match: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    totals = season_role_totals(player_match, shots)
    out = totals.copy()
    for q in ["npxg", *SUMMED]:
        out[q] = totals[q] / totals["minutes"] * 90
    return out
