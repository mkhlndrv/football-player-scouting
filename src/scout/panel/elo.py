import numpy as np
import pandas as pd

from scout.data import clubelo

# notebook 02 Step 3: opponent Elo on the match date covers 100% of Big-5 team-matches. The
# per-player slope on it is noise (y2y r 0.01-0.10) and never enters contribution; the join is
# kept for the role-average schedule effect and the report's coverage bar.
MATCH_KEYS = ["competition_id", "season", "game_id"]


def elo_on_dates(history: pd.DataFrame, dates: pd.Series) -> np.ndarray:
    """Rating from the interval containing each date; NaN outside the history. Never reads a
    later interval."""
    out = np.full(len(dates), np.nan)
    if history.empty:
        return out
    starts, ends, elos = (
        history["From"].to_numpy(),
        history["To"].to_numpy(),
        history["Elo"].to_numpy(),
    )
    when = pd.to_datetime(dates).dt.normalize().to_numpy()
    pos = np.searchsorted(starts, when, side="right") - 1
    inside = (pos >= 0) & (when <= ends[np.clip(pos, 0, len(ends) - 1)])
    out[inside] = elos[pos[inside]]
    return out


def club_elo_names(comps: list[str], seasons: list[int]) -> pd.Series:
    """Transfermarkt club_id -> ClubElo name, from the resolved lineage."""
    resolved = clubelo.resolved_clubs(comps, seasons)
    return resolved.drop_duplicates("club_id").set_index("club_id")["team_name"]


def opponent_elo(
    team_matches: pd.DataFrame, team_club: pd.DataFrame, elo_names: pd.Series
) -> pd.DataFrame:
    """One row per (match, team) with the opponent's Elo on the match date. `team_matches` has
    MATCH_KEYS, team_id, date; `team_club` maps (competition_id, team_id) -> club_id."""
    sides = team_matches[MATCH_KEYS + ["team_id", "date"]]
    pairs = sides.merge(
        sides[MATCH_KEYS + ["team_id"]].rename(columns={"team_id": "opp_id"}), on=MATCH_KEYS
    )
    pairs = pairs[pairs["team_id"] != pairs["opp_id"]]
    clubs = team_club.rename(columns={"team_id": "opp_id", "club_id": "opp_club_id"})
    pairs = pairs.merge(clubs, on=["competition_id", "opp_id"], how="left")
    pairs["opp_elo_name"] = pairs["opp_club_id"].map(elo_names)
    pairs["opp_elo"] = np.nan
    for name, idx in pairs.groupby("opp_elo_name").indices.items():
        pairs.iloc[idx, pairs.columns.get_loc("opp_elo")] = elo_on_dates(
            clubelo.fetch_club(name), pairs["date"].iloc[idx]
        )
    return pairs[MATCH_KEYS + ["team_id", "opp_id", "opp_elo"]]
