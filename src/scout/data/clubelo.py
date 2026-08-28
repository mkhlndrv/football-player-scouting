from io import StringIO

import numpy as np
import pandas as pd

from scout.data.http import polite_get

API = "http://api.clubelo.com/{key}"
COLUMNS = ["Rank", "Club", "Country", "Level", "Elo", "From", "To"]


def _csv(key: str) -> pd.DataFrame:
    response = polite_get(
        API.format(key=key), timeout_s=300
    )  # api.clubelo.com: 60-240 s per answer
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    if frame.empty:  # an unknown key answers 200 with the header only (notebook 01, Part 7b)
        return pd.DataFrame(columns=COLUMNS)
    frame["From"] = pd.to_datetime(frame["From"])
    frame["To"] = pd.to_datetime(frame["To"])
    return frame


def club_key(name: str) -> str:
    return name.replace(" ", "")  # ClubElo keys are display names without spaces


def fetch_club(name: str) -> pd.DataFrame:
    """Full rating history of one club as intervals; empty when ClubElo has no such club."""
    return _csv(club_key(name))


def list_clubs_on(date: str) -> pd.DataFrame:
    return _csv(date)


def elo_on(history: pd.DataFrame, date) -> float:
    if history.empty:
        return np.nan
    date = pd.Timestamp(date)
    hit = history[(history["From"] <= date) & (history["To"] >= date)]
    return float(hit["Elo"].iloc[0]) if len(hit) else np.nan


ELO_COUNTRY = {
    "ENG": "GB1",
    "ESP": "ES1",
    "ITA": "IT1",
    "GER": "L1",
    "FRA": "FR1",
    "BEL": "BE1",
    "NED": "NL1",
    "POR": "PO1",
    "TUR": "TR1",
    "SUI": "C1",
    "AUT": "A1",
    "DEN": "DK1",
}  # Brazil is not rated (notebook 01, Part 7b)


def resolved_clubs(comps: list[str], seasons: list[int]) -> pd.DataFrame:
    """ClubElo club name -> Transfermarkt club_id for every club rated on a 1 July of the panel
    (one date listing per season), through the team lineage and its overrides. Big 5 first."""
    from scout import config
    from scout.data import transfermarkt as tm
    from scout.identity import build_team_lineage, load_overrides

    listings = pd.concat(
        [list_clubs_on(f"{season}-07-01") for season in seasons], ignore_index=True
    )
    rated = listings[listings["Country"].isin(ELO_COUNTRY)]
    names = rated.assign(competition_id=rated["Country"].map(ELO_COUNTRY))
    names = (
        names[["competition_id", "Club"]].drop_duplicates().rename(columns={"Club": "team_name"})
    )
    clubs = tm.load_player_club_seasons(comps, seasons)
    clubs = clubs[["club_id", "club_name", "competition_id"]].drop_duplicates()
    lineage = build_team_lineage(clubs, {"clubelo": names}, load_overrides("teams"))
    lineage = lineage.dropna(subset=["club_id"])
    lineage["tier"] = (~lineage["competition_id"].isin(config.BIG5)).astype(int)
    return lineage.sort_values(["tier", "team_name"])[["team_name", "club_id", "competition_id"]]


def pull_histories(names: list[str], workers: int = 3, log=print) -> int:
    """Cache the rating history of every club; the host answers in 60-120 s and 502s under
    load, so a few requests in flight divide the wall time. Returns the non-empty count."""
    from concurrent.futures import ThreadPoolExecutor

    from scout.data import retry

    def one(name: str) -> bool:
        history = retry.until_done(lambda: fetch_club(name), log=log)
        return not history.empty

    found = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, ok in enumerate(pool.map(one, names), start=1):
            found += ok
            if i % 20 == 0:
                log(f"clubelo histories: {i}/{len(names)} | with history: {found}")
    return found
