import numpy as np
import pandas as pd
import pytest

from scout.models.intervals import bootstrap_sd, inflation, interval, role_prior, shrink


def test_shrink_moves_noisy_seasons_further_toward_the_mean():
    per90 = pd.Series([0.6, 0.6])
    out = shrink(per90, pd.Series([0.05, 0.2]), mu=0.3, tau2=0.01)
    assert out.point.iloc[0] > out.point.iloc[1] > 0.3
    assert out.k.iloc[0] > out.k.iloc[1]
    assert out.predictive_sd.iloc[1] > out.predictive_sd.iloc[0]


def test_inflation_makes_80pct_of_realised_fall_inside():
    rng = np.random.default_rng(1)
    point, sd = pd.Series(np.zeros(2000)), pd.Series(np.ones(2000))
    realised = pd.Series(rng.normal(0, 2.0, 2000))  # twice as spread as the stated sd
    f = inflation(point, sd, realised)
    assert f == pytest.approx(2.0, abs=0.15)
    band = interval(point, sd, f)
    assert ((realised >= band.lo) & (realised <= band.hi)).mean() == pytest.approx(0.8, abs=0.02)


def test_role_prior_and_bootstrap_sd_shapes():
    pairs = pd.DataFrame({"per90": [0.1, 0.2, 0.3, 0.4], "next": [0.1, 0.25, 0.3, 0.45]})
    mu, tau2 = role_prior(pairs)
    assert mu == pytest.approx(0.25) and tau2 > 0
    per_match = pd.DataFrame({"pid": [1] * 4, "out": [0.1, 0.4, 0.0, 0.3], "minutes": [90] * 4})
    sd = bootstrap_sd(per_match, ["pid"], "out", n_boot=50)
    assert sd.index.tolist() == [1] and sd.iloc[0] > 0
