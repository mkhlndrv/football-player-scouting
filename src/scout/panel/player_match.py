import pandas as pd

from scout.data import understat

# notebook 01 Step 5a: Understat codes are formation slots; six roles pass 20 players x 900 min
# in every league-season, AMC folds into W on its per-90 profile
ROLE_OF_CODE = {
    "GK": "GK",
    "DC": "CB",
    "DL": "FB",
    "DR": "FB",
    "DML": "FB",
    "DMR": "FB",
    "DMC": "CM",
    "MC": "CM",
    "ML": "W",
    "MR": "W",
    "AML": "W",
    "AMR": "W",
    "FWL": "W",
    "FWR": "W",
    "AMC": "W",
    "FW": "ST",
}
SUB_CODE = "Sub"
ROLES = ("GK", "CB", "FB", "CM", "W", "ST")
STINT_KEYS = ["league", "season", "team_id", "player_id"]


def assign_roles(rows: pd.DataFrame) -> pd.DataFrame:
    """`role` from the per-match code; substitutes take the player's main role (most starter
    minutes) at that club that season and stay NaN when they never started. `side` is the
    L/R suffix of the code."""
    unknown = set(rows["position"]) - set(ROLE_OF_CODE) - {SUB_CODE}
    if unknown:
        raise ValueError(f"unmapped Understat position codes: {sorted(unknown)}")
    out = rows.copy()
    out["is_sub"] = out["position"] == SUB_CODE
    out["role"] = out["position"].map(ROLE_OF_CODE)
    out["side"] = out["position"].str.extract(r"([LR])$")[0]

    starter_minutes = out[~out["is_sub"]].groupby(STINT_KEYS + ["role"])["minutes"].sum()
    main_role = (
        starter_minutes.reset_index()
        .sort_values("minutes", ascending=False)
        .drop_duplicates(STINT_KEYS)
        .rename(columns={"role": "_main_role"})
        .drop(columns="minutes")
    )
    out = out.merge(main_role, on=STINT_KEYS, how="left")
    out["role"] = out["role"].fillna(out["_main_role"])
    return out.drop(columns="_main_role")


def build(leagues: list[str] | None = None, seasons: list[int] | None = None) -> pd.DataFrame:
    return assign_roles(understat.load("player_match", leagues, seasons))
