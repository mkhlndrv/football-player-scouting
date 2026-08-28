from io import StringIO

import numpy as np
import pandas as pd

from scout.data.http import polite_get

API = "http://api.clubelo.com/{key}"
COLUMNS = ["Rank", "Club", "Country", "Level", "Elo", "From", "To"]


def _csv(key: str) -> pd.DataFrame:
    response = polite_get(API.format(key=key), timeout_s=120)  # api.clubelo.com often needs >30 s
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
