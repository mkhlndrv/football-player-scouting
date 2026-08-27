from collections.abc import Callable
from functools import partial

import pandas as pd

from scout import config
from scout.data import retry
from scout.data.http import polite_get

RAW = config.RAW
DEEP_STATS = "https://www.fotmob.com/api/data/leagueseasondeepstats"
STATIC_STAT = "https://data.fotmob.com/stats/{league_id}/season/{season_id}/{stat}.json"
CALENDAR_YEAR_LEAGUES = {"BRA1"}
ROW_COLUMNS = {
    "ParticiantId": "fotmob_player_id",  # sic: FotMob's own spelling
    "ParticipantName": "player_name",
    "TeamId": "fotmob_team_id",
    "TeamName": "team_name",
    "StatValue": "stat_value",
    "SubStatValue": "sub_stat_value",
    "MinutesPlayed": "minutes_played",
    "MatchesPlayed": "matches_played",
    "Rank": "rank",
    "Positions": "positions",
}


def season_name(comp: str, season: int) -> str:
    return str(season) if comp in CALENDAR_YEAR_LEAGUES else f"{season}/{season + 1}"


def league_seasons_and_stats(league_id: int) -> tuple[dict[str, int], pd.DataFrame]:
    """The deep-stats endpoint lists a league's season ids and its stat vocabulary; the stat
    argument only has to be a known name."""
    response = polite_get(
        DEEP_STATS, params={"id": league_id, "season": "", "type": "players", "stat": "goals"}
    )
    response.raise_for_status()
    payload = response.json()
    seasons = {s["name"]: s["id"] for s in payload["seasons"]}
    stats = pd.DataFrame(payload["statsList"])
    return seasons, stats[[c for c in ("name", "title", "category") if c in stats.columns]]


def stat_table(league_id: int, season_id: int, stat: str) -> pd.DataFrame:
    response = polite_get(STATIC_STAT.format(league_id=league_id, season_id=season_id, stat=stat))
    if response.status_code == 404:  # derived stats may have no static file: not an error
        return pd.DataFrame(columns=["stat", *ROW_COLUMNS.values()])
    response.raise_for_status()
    rows = response.json()["TopLists"][0]["StatList"]
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=["stat", *ROW_COLUMNS.values()])
    table = table[[c for c in ROW_COLUMNS if c in table.columns]].rename(columns=ROW_COLUMNS)
    table.insert(0, "stat", stat)
    return table


def raw_path(comp: str, season: int):
    return RAW / "fotmob" / f"{comp}_{season}.parquet"


def pull_league(comp: str, seasons: list[int], log: Callable[[str], None] = print) -> None:
    """One long parquet per league-season: every stat list the league offers, stored raw.
    Absence semantics (per-90 list → NaN, totals list → 0) are a cleaning decision, not fetch."""
    todo = [season for season in seasons if not raw_path(comp, season).exists()]
    if not todo:
        return
    league_id = config.FOTMOB_LEAGUES[comp]
    season_ids, stats = retry.until_done(partial(league_seasons_and_stats, league_id), log=log)
    for season in todo:
        name = season_name(comp, season)
        if name not in season_ids:
            log(f"fotmob {comp} {season}: no such season on provider")
            continue
        tables = [
            retry.until_done(partial(stat_table, league_id, season_ids[name], stat), log=log)
            for stat in stats.name
        ]
        frame = pd.concat(tables, ignore_index=True)
        frame.insert(0, "season", season)
        frame.insert(0, "competition_id", comp)
        path = raw_path(comp, season)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        log(f"fotmob {comp} {season}: {len(stats)} stats, {len(frame)} rows")


def load() -> pd.DataFrame:
    files = sorted((RAW / "fotmob").glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
