import numpy as np
import pandas as pd

# notebook 03 Step 4: the year-to-year change in contribution by role and age band (the delta
# method). Applied to the shrunk point it ties "no change" on held-out error at every horizon
# (MAE 0.057 / 0.063 / 0.066 per 90 at 1 / 2 / 3 seasons) and carries the age direction the
# resale model needs; a player-level deviation from the curve makes every horizon worse.
AGE_EDGES = [16, 20, 22, 24, 26, 28, 30, 32, 34, 45]
AGE_LABELS = ["<=20", "21-22", "23-24", "25-26", "27-28", "29-30", "31-32", "33-34", "35+"]
HORIZONS = (1, 2, 3)


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_EDGES, labels=AGE_LABELS)


def role_curve(pairs: pd.DataFrame) -> pd.Series:
    """Mean change in `point` to the next season, per (role, age band). `pairs` has role, age,
    point, point_next for consecutive seasons of the same player."""
    delta = pairs["point_next"] - pairs["point"]
    return delta.groupby([pairs["role"], age_band(pairs["age"])], observed=True).mean()


def project(
    point: pd.Series, age: pd.Series, role: pd.Series, curve: pd.Series, horizon: int
) -> pd.Series:
    """Walk the role curve `horizon` seasons forward from the current point and age."""
    out, current_age = point.astype(float).copy(), age.astype(float).copy()
    for _ in range(horizon):
        bands = age_band(current_age)
        step = [curve.get((r, b), 0.0) for r, b in zip(role, bands, strict=True)]
        out = out + pd.Series(step, index=out.index).fillna(0.0)
        current_age = current_age + 1
    return out


def horizon_inflation(z_scores: pd.Series) -> float:
    """Factor on the Phase 2 80% half-width so that 80% of realised outcomes at this horizon fall
    inside; fitted on base seasons up to the calibration cut, reported on the rest."""
    return float(np.percentile(np.abs(z_scores.dropna()), 80) / 1.2816)
