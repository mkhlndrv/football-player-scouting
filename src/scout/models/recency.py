import numpy as np
import pandas as pd

# notebook 02 Step 4: pooling three seasons beats last-season-only (r 0.899 vs 0.876 on 5,228
# player-role cases); half-lives 1-3 are within 0.001 of each other, 1.5 has the lowest rmse
HALF_LIFE = 1.5
SEASONS_BACK = 3


def weights(half_life: float = HALF_LIFE, seasons_back: int = SEASONS_BACK) -> np.ndarray:
    """Weights for lags 1..seasons_back (most recent first), summing to one."""
    raw = 0.5 ** (np.arange(seasons_back) / half_life)
    return raw / raw.sum()


def weighted_history(lagged: pd.DataFrame, half_life: float = HALF_LIFE) -> pd.Series:
    """`lagged` has one column per lag, most recent first (lag1, lag2, ...); rows with a
    missing lag are renormalised over the lags they have, so a two-season history still
    counts — it just leans more on what exists."""
    values = lagged.to_numpy(dtype=float)
    w = np.tile(weights(half_life, values.shape[1]), (len(values), 1))
    w[np.isnan(values)] = 0.0
    total = w.sum(axis=1)
    out = np.where(total > 0, np.nansum(values * w, axis=1) / np.where(total > 0, total, 1), np.nan)
    return pd.Series(out, index=lagged.index, name="weighted_history")
