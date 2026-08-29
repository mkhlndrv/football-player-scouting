import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# notebook 03 Step 5: next season's panel minutes from the last three seasons' minutes, age and
# role beat "last season's minutes" by 13% on held-out seasons (MAE 877 vs 1,006 minutes,
# dropouts counted as 0); injury history adds nothing (869.7 vs 869.6 on players with a file).
FORMULA = "target ~ lag1 + lag2 + lag3 + C(role) * (age + I(age**2))"
LAGS = ["lag1", "lag2", "lag3"]


def history(minutes: pd.DataFrame) -> pd.DataFrame:
    """One row per player and base season with the last three seasons' minutes (missing = 0),
    age and role. `minutes` has player_id, season, minutes, age, role."""
    wide = minutes.pivot_table(index="player_id", columns="season", values="minutes")
    ages = minutes.pivot_table(index="player_id", columns="season", values="age")
    roles = minutes.sort_values("minutes", ascending=False).drop_duplicates("player_id")
    roles = roles.set_index("player_id")["role"]
    rows = []
    for season in sorted(wide.columns):
        lags = {f"lag{i}": wide.get(season - i + 1) for i in (1, 2, 3)}
        block = pd.DataFrame(
            {k: (v if v is not None else np.nan) for k, v in lags.items()}, index=wide.index
        )
        block = block.dropna(subset=["lag1"]).fillna(0.0)
        block["age"] = (
            ages.get(season, pd.Series(np.nan, index=wide.index)).reindex(block.index) + 1
        )
        block["role"] = roles.reindex(block.index)
        block["season"] = season
        rows.append(block.reset_index())
    return pd.concat(rows, ignore_index=True).dropna(subset=["age", "role"])


def fit(train: pd.DataFrame):
    return smf.ols(FORMULA, data=train).fit()


def predict(model, rows: pd.DataFrame) -> pd.Series:
    return model.predict(rows).clip(lower=0.0)


def leave_future_out(frame: pd.DataFrame, first: int = 2019) -> pd.DataFrame:
    """`frame` = history rows with a `target` (next season's minutes, 0 when absent)."""
    out = []
    for season in sorted(frame["season"].unique()):
        if season < first:
            continue
        train, test = frame[frame["season"] < season], frame[frame["season"] == season]
        pred = predict(fit(train), test)
        out.append(
            {
                "season": int(season),
                "n": len(test),
                "mae_model": float((test["target"] - pred).abs().mean()),
                "mae_baseline": float((test["target"] - test["lag1"]).abs().mean()),
            }
        )
    return pd.DataFrame(out)
