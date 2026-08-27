import subprocess
from pathlib import Path

import duckdb
import pandas as pd

from scout.config import RAW, TM_DUCKDB_URL

DUCKDB_PATH = RAW / "transfermarkt.duckdb"


def ensure_duckdb(path: Path = DUCKDB_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", "-o", str(path), TM_DUCKDB_URL], check=True)
    return path


def connect(path: Path | None = None) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path or ensure_duckdb()), read_only=True)


def load_table(name: str, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    con = con or connect()
    return con.execute(f"SELECT * FROM {name}").df()


def load_player_club_seasons(
    comps: list[str], seasons: list[int], con: duckdb.DuckDBPyConnection | None = None
) -> pd.DataFrame:
    # appearances has whole club-seasons missing (notebook 01, Part 3); lineups cover every game,
    # so bench-only and gap rows come from lineups with minutes left NaN.
    con = con or connect()
    comp_list = ",".join(f"'{c}'" for c in comps)
    lo, hi = min(seasons), max(seasons)
    sql = f"""
    WITH apps AS (
      SELECT a.player_id, CAST(a.player_club_id AS INTEGER) AS club_id,
             CAST(g.season AS INTEGER) AS season, g.competition_id,
             SUM(a.minutes_played) AS minutes, COUNT(*) AS apps
      FROM appearances a JOIN games g ON CAST(a.game_id AS VARCHAR) = g.game_id
      WHERE g.competition_id IN ({comp_list}) AND CAST(g.season AS INTEGER) BETWEEN {lo} AND {hi}
      GROUP BY 1, 2, 3, 4
    ),
    lineups AS (
      SELECT DISTINCT l.player_id, l.club_id, CAST(g.season AS INTEGER) AS season, g.competition_id
      FROM game_lineups l JOIN games g ON CAST(l.game_id AS VARCHAR) = g.game_id
      WHERE g.competition_id IN ({comp_list}) AND CAST(g.season AS INTEGER) BETWEEN {lo} AND {hi}
    ),
    unioned AS (
      SELECT player_id, club_id, season, competition_id, minutes, apps, 'appearances' AS source
      FROM apps
      UNION ALL
      SELECT l.player_id, l.club_id, l.season, l.competition_id, NULL, NULL, 'lineups'
      FROM lineups l LEFT JOIN apps a
        ON a.player_id = l.player_id AND a.club_id = l.club_id AND a.season = l.season
      WHERE a.player_id IS NULL
    )
    SELECT u.player_id AS tm_player_id, p.name, p.date_of_birth, p.position, p.sub_position,
           u.club_id, c.name AS club_name, u.competition_id, u.season, u.minutes, u.apps, u.source
    FROM unioned u
    JOIN players p ON u.player_id = p.player_id
    LEFT JOIN clubs c ON CAST(c.club_id AS INTEGER) = u.club_id
    """
    return con.execute(sql).df()
