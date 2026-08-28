import duckdb
import pandas as pd

from scout.data import transfermarkt as tm
from scout.panel import freeze
from scout.panel.values import value_at

# notebook 01 Step 5b: no minutes floor — lineup-only rows stay with NaN minutes; the value is
# read both at the stint's first match (what the move cost) and at 1 July (the season's value)


def appearance_dates(
    comps: list[str], seasons: list[int], con: duckdb.DuckDBPyConnection
) -> pd.DataFrame:
    comp_list = ",".join(f"'{c}'" for c in comps)
    lo, hi = min(seasons), max(seasons)
    return con.execute(f"""
        SELECT a.player_id AS tm_player_id, CAST(a.player_club_id AS INTEGER) AS club_id,
               CAST(g.season AS INTEGER) AS season,
               MIN(a.date) AS first_date, MAX(a.date) AS last_date
        FROM appearances a JOIN games g ON a.game_id = g.game_id
        WHERE g.competition_id IN ({comp_list}) AND CAST(g.season AS INTEGER) BETWEEN {lo} AND {hi}
        GROUP BY 1, 2, 3
    """).df()


def add_values(stints: pd.DataFrame, valuations: pd.DataFrame) -> pd.DataFrame:
    out = stints.rename(columns={"tm_player_id": "player_id"})
    out["july_first"] = pd.to_datetime(out["season"].astype(str) + "-07-01")
    at_start = value_at(out, "first_date", valuations)
    at_july = value_at(out, "july_first", valuations)
    out["value_at_start"] = at_start["value"]
    out["value_age_days_start"] = at_start["value_age_days"]
    out["value_july"] = at_july["value"]
    out["value_age_days_july"] = at_july["value_age_days"]
    return out.rename(columns={"player_id": "tm_player_id"}).drop(columns="july_first")


def build(
    comps: list[str],
    seasons: list[int],
    con: duckdb.DuckDBPyConnection | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if as_of is not None:
        freeze.refuse_after(seasons, as_of)
    con = con or tm.connect()
    panel = tm.load_player_club_seasons(comps, seasons, con)
    dates = appearance_dates(comps, seasons, con)
    stints = panel.merge(dates, on=["tm_player_id", "club_id", "season"], how="left")
    valuations = con.execute(
        "SELECT player_id, date, market_value_in_eur FROM player_valuations"
    ).df()
    if as_of is not None:
        valuations = freeze.cut(valuations, as_of, date_col="date")
    return add_values(stints, valuations)
