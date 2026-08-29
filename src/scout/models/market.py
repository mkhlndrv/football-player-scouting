import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# notebook 03 Step 2: monotone gradient boosting beats the OLS regression on held-out log value in
# 8 of 9 seasons (rmse 0.185 vs 0.194 log10; median euro error 24% vs 28%); both far beat the
# market's own prior alone (0.333). Target = log10 value at the 1 July after the stats season.
NUMERIC = ["point", "history_point", "minutes", "age", "elo_c", "log_prior"]
CATEGORICAL = ["role", "competition_id"]
MONOTONE = [1, 1, 1, 0, 1, 1, 0, 0]  # value rises with everything but age, which is free
ROLES = ["CB", "CM", "FB", "ST", "W"]
ELO_CENTRE = 1600.0


def features(frame: pd.DataFrame, leagues: list[str]) -> pd.DataFrame:
    """The design matrix: numeric columns as they are, role and league as fixed category codes
    so a fitted model reads new rows the same way."""
    X = frame[NUMERIC].copy()
    X["role"] = pd.Categorical(frame["role"], categories=ROLES).codes
    X["competition_id"] = pd.Categorical(frame["competition_id"], categories=leagues).codes
    return X


def prepare(rows: pd.DataFrame) -> pd.DataFrame:
    """From contribution rows joined to valuations: the target and derived features. Expects
    value_july, value_next_july, club_elo_next, point, history_point, minutes, age, role,
    competition_id."""
    out = rows.dropna(subset=["value_next_july", "value_july", "club_elo_next"]).copy()
    out["y"] = np.log10(out["value_next_july"])
    out["log_prior"] = np.log10(out["value_july"])
    out["history_point"] = out["history_point"].fillna(out["point"])
    out["elo_c"] = (out["club_elo_next"] - ELO_CENTRE) / 100
    return out


def fit(train: pd.DataFrame, leagues: list[str], seed: int = 0) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        monotonic_cst=MONOTONE,
        categorical_features=[len(NUMERIC), len(NUMERIC) + 1],
        random_state=seed,
    )
    return model.fit(features(train, leagues), train["y"])


def leave_future_out(prepared: pd.DataFrame, leagues: list[str], first: int = 2016) -> pd.DataFrame:
    """One row per held-out season: trained on seasons before it, error on it."""
    rows = []
    for season in sorted(prepared["season"].unique()):
        if season < first:
            continue
        train, test = prepared[prepared["season"] < season], prepared[prepared["season"] == season]
        pred = fit(train, leagues).predict(features(test, leagues))
        err = test["y"] - pred
        rows.append(
            {
                "season": int(season),
                "n": len(test),
                "rmse": float(np.sqrt((err**2).mean())),
                "mae": float(err.abs().mean()),
                "mdape_eur": float(np.median(np.abs(10**pred / test["value_next_july"] - 1))),
            }
        )
    return pd.DataFrame(rows)
