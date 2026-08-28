from collections.abc import Callable
from functools import partial
from io import StringIO

import pandas as pd

from scout import config
from scout.data import retry
from scout.data.http import polite_get

RAW = config.RAW
COLUMNS = ["season", "injury", "from_date", "until_date", "days", "games_missed"]
PAGE_SIZE = 15  # Transfermarkt paginates the injury table (notebook 01, Part 7b)
MAX_PAGES = 12


def page_url(profile_url: str, page: int) -> str:
    base = profile_url.replace("/profil/", "/verletzungen/")
    return base if page == 1 else f"{base}/page/{page}"


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def parse_page(html: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return pd.DataFrame(columns=COLUMNS)
    table = next((t for t in tables if any("injury" in str(c).lower() for c in t.columns)), None)
    if table is None:
        return pd.DataFrame(columns=COLUMNS)
    table.columns = [str(c).strip().lower() for c in table.columns]
    return pd.DataFrame(
        {
            "season": table["season"].astype(str),
            "injury": table["injury"].astype(str),
            "from_date": pd.to_datetime(table["from"], format="%d/%m/%Y", errors="coerce"),
            "until_date": pd.to_datetime(table["until"], format="%d/%m/%Y", errors="coerce"),
            "days": _number(table["days"]),
            "games_missed": _number(table["games missed"]),  # '-' (no games) -> NaN
        }
    )


def fetch_injuries(profile_url: str) -> pd.DataFrame:
    """All spells of one player: pages are walked until one is empty or repeats the last."""
    pages: list[pd.DataFrame] = []
    for page in range(1, MAX_PAGES + 1):
        response = polite_get(page_url(profile_url, page))
        response.raise_for_status()
        spells = parse_page(response.text)
        if spells.empty or (pages and spells.equals(pages[-1])):
            break
        pages.append(spells)
    return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame(columns=COLUMNS)


def raw_path():
    return RAW / "injuries" / "spells.parquet"


def _append(batch: list[pd.DataFrame]) -> None:
    path = raw_path()
    new = pd.concat(batch, ignore_index=True)
    frame = pd.concat([pd.read_parquet(path), new], ignore_index=True) if path.exists() else new
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def pull_all(players: pd.DataFrame, log: Callable[[str], None] = print) -> None:
    """players: tm_player_id, url. A player with no spells gets one marker row (all NaN), so
    'fetched, none' is distinguishable from 'never fetched'. Appends every 50 players."""
    done = set(pd.read_parquet(raw_path()).tm_player_id) if raw_path().exists() else set()
    batch: list[pd.DataFrame] = []
    todo = players[~players.tm_player_id.isin(done)]
    for i, record in enumerate(todo.itertuples(index=False), start=1):
        spells = retry.until_done(partial(fetch_injuries, record.url), log=log)
        if spells.empty:
            spells = pd.DataFrame([{column: pd.NA for column in COLUMNS}])
        spells.insert(0, "tm_player_id", record.tm_player_id)
        batch.append(spells)
        if len(batch) >= 50:
            _append(batch)
            batch = []
            log(f"injuries: {i}/{len(todo)} players")
    if batch:
        _append(batch)


def load() -> pd.DataFrame:
    path = raw_path()
    return (
        pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["tm_player_id", *COLUMNS])
    )
