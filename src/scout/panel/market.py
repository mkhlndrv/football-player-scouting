import duckdb
import numpy as np
import pandas as pd

from scout.data import transfermarkt as tm
from scout.panel import freeze

MIRROR_MONTHS = (
    36  # notebook 01 Part 2d: 0.22% of loans run past 36 months; 48 sweeps in re-purchases
)
KINDS = ("loan_return", "loan_out", "paid", "free", "undisclosed", "internal")

# fee-independent: a move mirrored by the reverse move is a loan whatever the fee says (paid loans
# exist); NULL fees between two clubs the dataset knows are undisclosed, otherwise youth/reserve/
# without-club moves
_CLASSIFY = f"""
WITH moves AS (
  SELECT player_id, transfer_date, transfer_season, from_club_id, to_club_id,
         from_club_name, to_club_name, transfer_fee, market_value_in_eur
  FROM transfers
  WHERE from_club_id IS NOT NULL AND to_club_id IS NOT NULL
    AND TRY_CAST(SUBSTR(transfer_season, 1, 2) AS INTEGER) BETWEEN {{lo}} AND {{hi}}
),
known AS (SELECT DISTINCT CAST(club_id AS INTEGER) AS club_id FROM clubs),
flags AS (
  SELECT m.*,
    EXISTS (SELECT 1 FROM moves r WHERE r.player_id = m.player_id
            AND r.from_club_id = m.to_club_id AND r.to_club_id = m.from_club_id
            AND r.transfer_date < m.transfer_date
            AND r.transfer_date >= m.transfer_date - INTERVAL {MIRROR_MONTHS} MONTH) AS is_return,
    EXISTS (SELECT 1 FROM moves r WHERE r.player_id = m.player_id
            AND r.from_club_id = m.to_club_id AND r.to_club_id = m.from_club_id
            AND r.transfer_date > m.transfer_date
            AND r.transfer_date <= m.transfer_date + INTERVAL {MIRROR_MONTHS} MONTH)
      AS mirror_after,
    m.from_club_id IN (SELECT club_id FROM known)
      AND m.to_club_id IN (SELECT club_id FROM known) AS both_known
  FROM moves m
)
SELECT player_id, transfer_date, transfer_season, from_club_id, to_club_id, from_club_name,
       to_club_name, transfer_fee, market_value_in_eur,
  CASE WHEN is_return THEN 'loan_return'
       WHEN mirror_after THEN 'loan_out'
       WHEN transfer_fee > 0 THEN 'paid'
       WHEN transfer_fee = 0 THEN 'free'
       WHEN both_known THEN 'undisclosed'
       ELSE 'internal' END AS kind
FROM flags
"""


def classify(con: duckdb.DuckDBPyConnection, seasons: tuple[int, int] = (14, 25)) -> pd.DataFrame:
    return con.execute(_CLASSIFY.format(lo=seasons[0], hi=seasons[1])).df()


def add_cost(moves: pd.DataFrame) -> pd.DataFrame:
    """Spec 4.I: what the buying club paid — the fee when disclosed, the market value at the
    move for free and undisclosed transfers; loans and internal moves are not purchases."""
    out = moves.copy()
    fee = pd.to_numeric(out["transfer_fee"]).astype(float)
    value = pd.to_numeric(out["market_value_in_eur"]).astype(float)
    out["cost"] = np.select(
        [out["kind"] == "paid", out["kind"].isin(["free", "undisclosed"])], [fee, value], np.nan
    )
    return out


def build(
    con: duckdb.DuckDBPyConnection | None = None, as_of: pd.Timestamp | None = None
) -> pd.DataFrame:
    moves = classify(con or tm.connect())
    if as_of is not None:
        moves = freeze.cut(moves, as_of, date_col="transfer_date")
    return add_cost(moves)
