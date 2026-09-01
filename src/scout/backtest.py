"""Phase 5 backtest core: orderings, shortlist scoring and paired verdicts (notebook 05)."""

import numpy as np
import pandas as pd

FIT_ROLES = {"CB", "FB", "CM"}
OUT_ROLES = {"W", "ST", "GK"}
FIT_ORDERING = "f3"  # rank-mean(similarity to X, expected minutes) — Step 4
OUTPUT_ORDERING = "o2"  # P(>= bar) within budget — Step 4
GATE_K = 50  # top-k most similar within role — Step 2
SHORTLIST = 5
OUTCOME_COLUMNS = ["minutes_per_meur", "ga90_mean", "value_ratio"]
QUALITY_SIGNS = {  # Step 5c: stats that passed stability + travel, sign = "higher is better"
    "aerialDuelsWonPercentage": 1,
    "groundDuelsWonPercentage": 1,
    "dribbledPast": -1,
    "possessionLost": -1,
}


def ordering_key(pool: pd.DataFrame, ordering: str) -> pd.Series:
    """Lower key = earlier on the shortlist. NaN key = ineligible for this ordering."""
    if ordering == "f1":
        return pool["dist"].rank()
    if ordering == "f2":
        return (pool["dist"].rank() + pool["value"].rank()) / 2
    if ordering == "f3":
        return (pool["dist"].rank() + (-pool["expected_minutes"]).rank()) / 2
    if ordering == "prod_per_eur":  # Step 5g/5h: surplus production per euro (spec objective)
        return (-pool["prod_per_eur"]).rank()
    if ordering == "blend":  # Step 5d: equal rank-mix of similarity and P(>= bar)
        return (pool["dist"].rank() + (-pool["p_bar"]).rank()) / 2
    if ordering == "o1":
        return (-(pool["p_bar"] / (pool["value"] / 1e6))).rank()
    if ordering == "o2":
        return (-pool["p_bar"]).rank()
    if ordering == "defence":
        return (-pool["def_z"]).rank()
    if ordering == "quality":
        return (-pool["qual_z"]).rank()
    if ordering == "market":
        return (-pool["value"]).rank()
    if ordering == "naive":
        return (-pool["ga90"]).rank()
    raise ValueError(f"unknown ordering {ordering!r}")


def shortlist(pool: pd.DataFrame, ordering: str, n: int = SHORTLIST) -> pd.DataFrame:
    return pool.assign(_k=ordering_key(pool, ordering)).dropna(subset=["_k"]).nsmallest(n, "_k")


def score_case(top: pd.DataFrame) -> dict:
    scores = {
        "minutes_per_meur": top["out_minutes"].fillna(0).sum() / (top["value"].sum() / 1e6),
        "ga90_mean": top["out_ga90"].mean(),
        "value_ratio": top["out_value_next"].sum() / top["value"].sum(),
    }
    if "out_duel_z" in top:
        # Reported for the defensive roles, never part of the majority verdict: the three
        # columns above are the graded ones and must stay comparable across the tournament.
        scores["duel_quality"] = top["out_duel_z"].mean()
    return scores


def scores(pools: pd.DataFrame, ordering: str, n: int = SHORTLIST) -> pd.DataFrame:
    """Per-case outcome scores of the top-n under an ordering; cases with a short list drop."""
    rows = []
    for case_id, group in pools.groupby("case_id"):
        top = shortlist(group, ordering, n)
        if len(top) < n:
            continue
        rows.append({"case_id": case_id, **score_case(top)})
    if not rows:
        return pd.DataFrame(columns=OUTCOME_COLUMNS)
    return pd.DataFrame(rows).set_index("case_id")


def quality_z(pools: pd.DataFrame) -> pd.Series:
    """Equal-z defender-quality score within (dep_role, transfer_season) — Step 5c."""
    parts = []
    for column, sign in QUALITY_SIGNS.items():
        z = pools.groupby(["dep_role", "transfer_season"])[column].transform(
            lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1.0)
        )
        parts.append(sign * z)
    return pd.concat(parts, axis=1).mean(axis=1)


def case_wins(a: pd.DataFrame, b: pd.DataFrame, ids: pd.Index) -> tuple[pd.DataFrame, pd.Series]:
    """Per-column wins of a over b, and the majority (>= 2 of 3) verdict per case."""
    both = a.loc[ids, OUTCOME_COLUMNS].join(b.loc[ids, OUTCOME_COLUMNS], lsuffix="_a", rsuffix="_b")
    columns = pd.DataFrame(
        {
            c: (both[f"{c}_a"] > both[f"{c}_b"]).where(
                both[f"{c}_a"].notna() & both[f"{c}_b"].notna()
            )
            for c in OUTCOME_COLUMNS
        }
    )
    majority = columns.sum(axis=1, min_count=1) >= 2
    return columns, majority[columns.notna().any(axis=1)]


def verdict(a: pd.DataFrame, b: pd.DataFrame, ids: pd.Index, seed: int = 0) -> dict:
    """Paired case-win rate of a over b with a bootstrap 80% interval over cases."""
    ids = a.index.intersection(b.index).intersection(ids)
    if len(ids) < 5:
        return {"n": int(len(ids)), "too_few": True}
    columns, majority = case_wins(a, b, ids)
    wins = majority.to_numpy(float)
    rng = np.random.default_rng(seed)
    boots = [wins[rng.integers(0, len(wins), len(wins))].mean() for _ in range(1000)]
    return {
        "n": int(len(ids)),
        "case_win": round(float(majority.mean()), 4),
        "lo": round(float(np.percentile(boots, 10)), 4),
        "hi": round(float(np.percentile(boots, 90)), 4),
        "columns": {c: round(float(columns[c].mean()), 4) for c in OUTCOME_COLUMNS},
    }
