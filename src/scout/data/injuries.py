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


PANEL_START = pd.Timestamp("2014-07-01")  # spells older than the panel are never used


def fetch_injuries(profile_url: str, since: pd.Timestamp = PANEL_START) -> pd.DataFrame:
    """Spells of one player, newest page first; pages are walked until one is empty, repeats
    the last, or has already reached spells older than `since`."""
    pages: list[pd.DataFrame] = []
    for page in range(1, MAX_PAGES + 1):
        response = polite_get(page_url(profile_url, page))
        response.raise_for_status()
        spells = parse_page(response.text)
        if spells.empty or (pages and spells.equals(pages[-1])):
            break
        pages.append(spells)
        if spells["from_date"].min() < since:
            break
    return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame(columns=COLUMNS)


def raw_path():
    return RAW / "injuries" / "spells.parquet"


def _append(batch: list[pd.DataFrame]) -> None:
    path = raw_path()
    new = pd.concat(batch, ignore_index=True)
    frame = pd.concat([pd.read_parquet(path), new], ignore_index=True) if path.exists() else new
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def panel_players(comps: list[str], seasons: list[int]) -> pd.DataFrame:
    """tm_player_id, url for every panel player — Big-5 players first (5e and the backtest need
    them), then feeder-league players, most minutes first within each tier."""
    from scout import config
    from scout.data import transfermarkt as tm

    panel = tm.load_player_club_seasons(comps, seasons)
    panel["tier"] = (~panel.competition_id.isin(config.BIG5)).astype(int)
    order = panel.groupby("tm_player_id").agg(tier=("tier", "min"), minutes=("minutes", "sum"))
    order = order.sort_values(["tier", "minutes"], ascending=[True, False])
    urls = tm.load_table("players")[["player_id", "url"]].rename(
        columns={"player_id": "tm_player_id"}
    )
    return order.reset_index()[["tm_player_id"]].merge(urls, on="tm_player_id")


def pull_all(players: pd.DataFrame, log: Callable[[str], None] = print) -> None:
    """players: tm_player_id, url, in pull order. A player with no spells gets one marker row
    (all NaN), so 'fetched, none' is distinguishable from 'never fetched'. Appends every 50."""
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
