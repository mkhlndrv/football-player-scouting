import numpy as np
import pandas as pd

# notebook 02 Step 7: factors are median ratios of xG+xA per 90 after / before a move, from
# movers with a real output before it; the ratio form beats the difference form on held-out
# movers in every tier pair, and the Big-5 -> Big-5 factor (0.85) is regression to the mean
# that every pair carries. FotMob totals are absent for 40% of player-seasons: absent is NaN.
MIN_OUTPUT_BEFORE = 0.10
MIN_MOVERS = 30
OUTPUT_FLOOR = 0.02  # keeps a zero season after the move finite in the log
AGE_BANDS = [15, 21, 24, 27, 30, 45]
AGE_LABELS = ["<=21", "22-24", "25-27", "28-30", "31+"]


def log_ratios(movers: pd.DataFrame) -> pd.Series:
    """Age-adjusted log(after / before); `movers` has output, output_after, age."""
    kept = movers[movers["output"] >= MIN_OUTPUT_BEFORE]
    raw = np.log(kept["output_after"].clip(lower=OUTPUT_FLOOR) / kept["output"])
    bands = pd.cut(kept["age"], AGE_BANDS, labels=AGE_LABELS)
    effect = raw.groupby(bands, observed=True).mean()
    return (raw - bands.map(effect).astype(float) + raw.mean()).rename("log_ratio")


def factor(values: pd.Series, n_boot: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """Median multiplier with an 80% bootstrap interval on the median."""
    rng = np.random.default_rng(seed)
    x = values.to_numpy()
    boots = [np.median(rng.choice(x, len(x))) for _ in range(n_boot)]
    return (
        float(np.exp(np.median(x))),
        float(np.exp(np.percentile(boots, 10))),
        float(np.exp(np.percentile(boots, 90))),
    )


def individual_spread(values: pd.Series) -> tuple[float, float]:
    """10th and 90th percentile of individual log ratios around the median: the 80% interval
    for one mover is factor * exp(spread)."""
    centred = values - np.median(values)
    return float(np.percentile(centred, 10)), float(np.percentile(centred, 90))


def tier_factors(movers: pd.DataFrame, big5: set[str]) -> pd.DataFrame:
    """One row per (tier_from, tier_to) plus every league pair with >= MIN_MOVERS movers."""
    ratios = log_ratios(movers)
    kept = movers.loc[ratios.index].assign(log_ratio=ratios)
    kept["tier_from"] = np.where(kept["competition_id"].isin(big5), "big5", "feeder")
    kept["tier_to"] = np.where(kept["league_to"].isin(big5), "big5", "feeder")
    rows = []
    for keys in (["tier_from", "tier_to"], ["competition_id", "league_to"]):
        for names, group in kept.groupby(keys):
            if len(group) < MIN_MOVERS:
                continue
            f, lo, hi = factor(group["log_ratio"])
            s_lo, s_hi = individual_spread(group["log_ratio"])
            rows.append((*names, len(group), f, lo, hi, s_lo, s_hi))
    columns = ["from", "to", "movers", "factor", "p10", "p90", "spread_lo", "spread_hi"]
    return pd.DataFrame(rows, columns=columns)
