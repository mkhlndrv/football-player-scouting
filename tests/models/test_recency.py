import numpy as np
import pandas as pd
import pytest

from scout.models.recency import HALF_LIFE, weighted_history, weights


def test_weights_halve_every_half_life_and_sum_to_one():
    w = weights(half_life=1.0, seasons_back=3)
    assert w.sum() == pytest.approx(1.0)
    assert w[1] / w[0] == pytest.approx(0.5) and w[2] / w[1] == pytest.approx(0.5)
    assert HALF_LIFE == 1.5


def test_weighted_history_renormalises_over_available_lags():
    lagged = pd.DataFrame(
        {"lag1": [1.0, 1.0, np.nan], "lag2": [0.0, np.nan, np.nan], "lag3": [0.0, np.nan, np.nan]}
    )
    out = weighted_history(lagged, half_life=1.0)
    assert out.iloc[0] == pytest.approx(4 / 7)  # weights 4:2:1
    assert out.iloc[1] == pytest.approx(1.0)  # only lag1 present
    assert np.isnan(out.iloc[2])
