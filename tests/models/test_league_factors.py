import numpy as np
import pandas as pd
import pytest

from scout.models.league_factors import MIN_MOVERS, factor, log_ratios, tier_factors


def _movers(n, before, after, comp="NL1", to="GB1", seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "competition_id": [comp] * n,
            "league_to": [to] * n,
            "output": np.full(n, before),
            "output_after": after * np.exp(rng.normal(0, 0.05, n)),
            "age": rng.integers(20, 31, n),
        }
    )


def test_log_ratios_drop_movers_without_output_before():
    movers = pd.concat([_movers(3, 0.3, 0.2), _movers(2, 0.05, 0.2)], ignore_index=True)
    assert len(log_ratios(movers)) == 3


def test_factor_recovers_a_known_multiplier_with_an_interval():
    movers = _movers(200, 0.30, 0.20)
    f, lo, hi = factor(log_ratios(movers))
    assert f == pytest.approx(2 / 3, abs=0.03) and lo < f < hi


def test_tier_factors_pool_by_tier_and_list_large_pairs_only():
    movers = pd.concat(
        [_movers(40, 0.3, 0.2), _movers(10, 0.3, 0.25, comp="BE1")], ignore_index=True
    )
    table = tier_factors(movers, big5={"GB1"})
    assert set(zip(table["from"], table["to"], strict=True)) == {("feeder", "big5"), ("NL1", "GB1")}
    assert (table.movers >= MIN_MOVERS).all()
