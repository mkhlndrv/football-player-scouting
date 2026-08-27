from collections.abc import Callable
from functools import partial

import pandas as pd

from scout import config
from scout.data import retry
from scout.data.http import polite_get

RAW = config.RAW
API = "https://api.sofascore.com/api/v1"
HEADERS = {"Referer": "https://www.sofascore.com/", "Origin": "https://www.sofascore.com"}
POSITION_GROUPS = ("G", "D", "M", "F")  # the statistics endpoint exposes position only via filter
PAGE_SIZE = 100
# All 40 accepted by the endpoint (notebook 01, Part 5). Which become features is Step 5's call.
FIELDS = [
    "minutesPlayed",
    "appearances",
    "rating",
    "goals",
    "assists",
    "expectedGoals",
    "expectedAssists",
    "keyPasses",
    "bigChancesCreated",
    "accuratePasses",
    "accuratePassesPercentage",
    "accurateFinalThirdPasses",
    "accurateLongBalls",
    "accurateCrosses",
    "successfulDribbles",
    "tackles",
    "interceptions",
    "clearances",
    "ballRecovery",
    "possessionWonAttThird",
    "possessionLost",
    "groundDuelsWon",
    "groundDuelsWonPercentage",
    "aerialDuelsWon",
    "aerialDuelsWonPercentage",
    "dribbledPast",
    "fouls",
    "wasFouled",
    "errorLeadToShot",
    "errorLeadToGoal",
    "touches",
    "saves",
    "goalsConceded",
    "savedShotsFromInsideTheBox",
    "savedShotsFromOutsideTheBox",
    "highClaims",
    "punches",
    "penaltyFaced",
    "penaltySave",
    "cleanSheet",
]
CALENDAR_YEAR_LEAGUES = {"BRA1"}


def _get_json(path: str, params: dict | None = None) -> dict:
    response = polite_get(f"{API}{path}", params=params, headers=HEADERS, tls=True)
    response.raise_for_status()
    return response.json()


def list_seasons(tournament_id: int) -> pd.DataFrame:
    seasons = _get_json(f"/unique-tournament/{tournament_id}/seasons")["seasons"]
    return pd.DataFrame(
        {"season_id": [s["id"] for s in seasons], "year": [s["year"] for s in seasons]}
    )


def season_year(comp: str, season: int) -> str:
    return str(season) if comp in CALENDAR_YEAR_LEAGUES else config.season_short(season)


def position_group_stats(
    tournament_id: int, season_id: int, group: str, fields: list[str] = FIELDS
) -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        payload = _get_json(
            f"/unique-tournament/{tournament_id}/season/{season_id}/statistics",
            params={
                "limit": PAGE_SIZE,
                "offset": offset,
                "accumulation": "total",
                "fields": ",".join(fields),
                "filters": f"position.in.{group}",
            },
        )
        for result in payload.get("results", []):
            rows.append(
                {
                    "sofascore_player_id": result["player"]["id"],
                    "player_name": result["player"]["name"],
                    "position_group": group,
                    "sofascore_team_id": result["team"]["id"],
                    "team_name": result["team"]["name"],
                    **{field: result.get(field) for field in fields},
                }
            )
        if payload.get("page", 1) >= payload.get("pages", 1):
            break
        offset += PAGE_SIZE
    return pd.DataFrame(rows)


def raw_path(comp: str, season: int):
    return RAW / "sofascore" / f"{comp}_{season}.parquet"


def pull_league(comp: str, seasons: list[int], log: Callable[[str], None] = print) -> None:
    """One parquet per league-season (all four position groups); existing files are skipped."""
    tournament_id = config.SOFASCORE_TOURNAMENTS[comp]
    season_ids = list_seasons(tournament_id).set_index("year").season_id
    for season in seasons:
        path = raw_path(comp, season)
        if path.exists():
            continue
        year = season_year(comp, season)
        if year not in season_ids.index:
            log(f"sofascore {comp} {season}: no such season on provider")
            continue
        season_id = int(season_ids[year])
        groups = [
            retry.until_done(
                partial(position_group_stats, tournament_id, season_id, group), log=log
            )
            for group in POSITION_GROUPS
        ]
        frame = pd.concat(groups, ignore_index=True)
        frame.insert(0, "season", season)
        frame.insert(0, "competition_id", comp)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        log(f"sofascore {comp} {season}: {len(frame)} players")


def load() -> pd.DataFrame:
    files = sorted((RAW / "sofascore").glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
