import pandas as pd

# notebook 02 Step 8: the 20th percentile of regulars is stable season to season (CV 0.05-0.21
# per role) on 250-420 players; the median of cheap signings is not (CV 0.19-1.34 on 5-12) and
# sits at the 33rd-61st percentile of the role - a survivor, not the freely available level.
PERCENTILE = 0.20
KEYS = ["role", "season"]


def replacement_level(regulars: pd.DataFrame, value: str, keys: list[str] = KEYS) -> pd.DataFrame:
    """Per key, the PERCENTILE of `value` over players with >= MIN_MINUTES in the role-season,
    and the count behind it. Keepers pass their own measure as `value`."""
    grouped = regulars.groupby(keys)[value]
    out = grouped.quantile(PERCENTILE).rename("replacement_level").to_frame()
    out["players"] = grouped.size()
    return out.reset_index()


def surplus(
    frame: pd.DataFrame, value: str, levels: pd.DataFrame, keys: list[str] = KEYS
) -> pd.Series:
    """Contribution above the replacement level of the player's role-season."""
    merged = frame[keys + [value]].merge(levels[keys + ["replacement_level"]], on=keys, how="left")
    return (merged[value] - merged["replacement_level"]).set_axis(frame.index).rename("surplus")
