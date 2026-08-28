import pandas as pd

from scout.data import reep
from scout.identity import match_players, resolve_player_ids

# provider frame -> (id column, name column, team column, minutes column)
PROVIDER_COLUMNS = {
    "understat": ("player_id", "player", "team", "minutes"),
    "sofascore": ("sofascore_player_id", "player_name", "team_name", "minutesPlayed"),
    "fotmob": ("fotmob_player_id", "player_name", "team_name", "fm_minutes"),
}
KEYS = ["competition_id", "season"]


def transfermarkt_side(tm_panel: pd.DataFrame) -> pd.DataFrame:
    side = tm_panel.rename(columns={"tm_player_id": "right_id"})
    side = side[["right_id", "name", "club_id", "season", "competition_id"]].copy()
    side["club_key"] = side["club_id"].astype("Int64").astype(str)
    return side


def resolve_provider(
    provider: str,
    rows: pd.DataFrame,
    tm_side: pd.DataFrame,
    lineage: pd.DataFrame,
    people: pd.DataFrame,
) -> pd.DataFrame:
    """Every provider player-season with its Transfermarkt id (Step 4 policy: reep key, else the
    name cascade at 85 inside the club-season, modal id across seasons) and `source`."""
    id_col, name_col, team_col, minutes_col = PROVIDER_COLUMNS[provider]
    clubs = lineage[lineage["provider"] == provider][["competition_id", "team_name", "club_id"]]
    keyed = rows.merge(
        clubs.rename(columns={"team_name": team_col}), on=["competition_id", team_col], how="left"
    )
    keyed["club_key"] = keyed["club_id"].astype("Int64").astype(str)
    keyed = keyed.rename(columns={name_col: "name", id_col: "provider_id", minutes_col: "minutes"})

    matches = []
    for (comp, season), left in keyed.groupby(KEYS):
        right = tm_side[(tm_side["competition_id"] == comp) & (tm_side["season"] == season)]
        out = match_players(left[["name", "club_key", "season", "provider_id", "minutes"]], right)
        matches.append(out.assign(competition_id=comp))
    matches = pd.concat(matches, ignore_index=True)
    matches["provider_id"] = matches["provider_id"].astype(int).astype(str)
    resolved = resolve_player_ids(matches, reep.transfermarkt_keys(people, provider))
    return matches.merge(resolved, on="provider_id")


def minutes_rate(resolved: pd.DataFrame) -> pd.DataFrame:
    """Share of provider minutes carrying a Transfermarkt id, per competition and season."""
    bridged = resolved["minutes"].where(resolved["tm_player_id"].notna(), 0)
    grouped = resolved.assign(bridged=bridged).groupby(KEYS)
    return (grouped["bridged"].sum() / grouped["minutes"].sum()).unstack("season")
