import numpy as np
import pandas as pd

from scout.models.availability import fit, history, predict


def test_history_builds_lags_with_zero_for_missing_seasons():
    minutes = pd.DataFrame(
        {
            "player_id": [1, 1, 2],
            "season": [2020, 2022, 2022],
            "minutes": [900, 1800, 700],
            "age": [24, 26, 30],
            "role": ["W", "W", "CB"],
        }
    )
    h = history(minutes).set_index(["player_id", "season"])
    assert h.loc[(1, 2022), ["lag1", "lag2", "lag3"]].tolist() == [1800, 0, 900]
    assert h.loc[(1, 2022), "age"] == 27 and h.loc[(2, 2022), "role"] == "CB"


def test_fit_predicts_more_minutes_for_more_recent_minutes():
    rng = np.random.default_rng(0)
    n = 300
    lag1 = rng.uniform(600, 3400, n)
    train = pd.DataFrame(
        {
            "lag1": lag1,
            "lag2": lag1 * 0.9,
            "lag3": lag1 * 0.8,
            "age": rng.integers(20, 33, n),
            "role": rng.choice(["W", "CB"], n),
            "target": 0.8 * lag1 + rng.normal(0, 100, n),
        }
    )
    model = fit(train)
    low, high = train.iloc[[0]].assign(lag1=800), train.iloc[[0]].assign(lag1=3000)
    assert predict(model, high).iloc[0] > predict(model, low).iloc[0] >= 0
