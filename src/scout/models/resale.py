import numpy as np
import pandas as pd

from scout.models import market, trajectory

# notebook 03 Step 6: resale at horizon h chains the market model through the intermediate
# seasons (the expected value at h-1 is the prior for h) on the trajectory's projected point and
# age; the 80% band is empirical per horizon - the 10th/90th percentiles of realised minus
# predicted on earlier base seasons - because a Monte-Carlo over the two intervals does not
# compound the market residual (coverage 0.90 / 0.74 / 0.64 at horizons 1 / 2 / 3).
HORIZONS = trajectory.HORIZONS


def median_log_value(
    rows: pd.DataFrame, model, curve: pd.Series, leagues: list[str], horizon: int
) -> pd.Series:
    """Expected log10 value `horizon` seasons ahead for prepared market rows."""
    prior = rows["log_prior"].copy()
    for step in range(1, horizon + 1):
        sim = rows.copy()
        sim["point"] = trajectory.project(rows["point"], rows["age"], rows["role"], curve, step)
        sim["history_point"] = sim["point"]
        sim["age"] = rows["age"] + step
        sim["log_prior"] = prior
        prior = pd.Series(model.predict(market.features(sim, leagues)), index=rows.index)
    return prior


def empirical_bands(residuals: dict[int, pd.Series]) -> dict[int, tuple[float, float]]:
    """Per horizon, the 10th and 90th percentile of realised minus predicted log10 value."""
    return {
        h: tuple(float(x) for x in np.percentile(r.dropna(), [10, 90]))
        for h, r in residuals.items()
    }


def interval(median: pd.Series, band: tuple[float, float]) -> pd.DataFrame:
    return pd.DataFrame({"lo": median + band[0], "hi": median + band[1]})
