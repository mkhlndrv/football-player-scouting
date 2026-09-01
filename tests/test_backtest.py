import numpy as np
import pandas as pd
import pytest

from scout import backtest


def pool(**overrides):
    base = pd.DataFrame(
        {
            "case_id": ["c1"] * 4,
            "dist": [1.0, 2.0, 3.0, 4.0],
            "value": [10e6, 5e6, 20e6, 1e6],
            "expected_minutes": [1000.0, 3000.0, 2000.0, 500.0],
            "p_bar": [0.2, 0.8, 0.5, 0.9],
            "def_z": [0.5, np.nan, -1.0, 2.0],
            "qual_z": [0.1, 0.4, np.nan, -0.2],
            "ga90": [0.1, 0.6, 0.3, 0.2],
            "prod_per_eur": [0.5, 2.0, 1.0, 3.0],
            "out_minutes": [900.0, 2500.0, np.nan, 100.0],
            "out_ga90": [0.2, 0.5, 0.1, np.nan],
            "out_value_next": [12e6, 8e6, 15e6, 2e6],
        }
    )
    for key, val in overrides.items():
        base[key] = val
    return base


def test_ordering_keys_sort_as_designed():
    p = pool()
    assert backtest.shortlist(p, "f1", 2).index.tolist() == [0, 1]  # closest first
    assert backtest.shortlist(p, "o2", 2).p_bar.tolist() == [0.9, 0.8]  # highest P first
    assert backtest.shortlist(p, "market", 1).value.iloc[0] == 20e6  # priciest first
    assert backtest.shortlist(p, "naive", 1).ga90.iloc[0] == 0.6
    assert backtest.shortlist(p, "prod_per_eur", 1).index.tolist() == [3]
    # blend = rank-mean of dist asc and p_bar desc: row1 (dist 2, p 0.8) wins
    assert backtest.shortlist(pool(), "blend", 1).index.tolist() == [1]
    # f3 = rank-mean of dist asc and expected_minutes desc: row1 (dist 2, minutes 3000) wins
    assert backtest.shortlist(p, "f3", 1).index.tolist() == [1]


def test_nan_keys_are_ineligible():
    assert 1 not in backtest.shortlist(pool(), "defence", 4).index  # NaN def_z drops
    assert len(backtest.shortlist(pool(), "quality", 4)) == 3


def test_unknown_ordering_raises():
    with pytest.raises(ValueError):
        backtest.ordering_key(pool(), "vibes")


def test_score_case_arithmetic():
    top = pool().iloc[:2]
    s = backtest.score_case(top)
    assert s["minutes_per_meur"] == pytest.approx((900 + 2500) / 15.0)
    assert s["ga90_mean"] == pytest.approx(0.35)
    assert s["value_ratio"] == pytest.approx(20e6 / 15e6)


def test_scores_drops_short_cases():
    assert backtest.scores(pool(), "o2", n=5).empty  # only 4 candidates
    assert len(backtest.scores(pool(), "o2", n=3)) == 1


def test_case_wins_majority_and_nan_handling():
    a = pd.DataFrame(
        {"minutes_per_meur": [2.0], "ga90_mean": [0.5], "value_ratio": [np.nan]},
        index=["c1"],
    )
    b = pd.DataFrame(
        {"minutes_per_meur": [1.0], "ga90_mean": [0.1], "value_ratio": [1.0]},
        index=["c1"],
    )
    columns, majority = backtest.case_wins(a, b, a.index)
    assert bool(majority.loc["c1"]) is True  # wins 2 of the 2 comparable columns
    assert columns.value_ratio.isna().all()


def test_verdict_deterministic_and_bounded():
    rng = np.random.default_rng(1)
    a = pd.DataFrame(
        {c: rng.random(40) for c in backtest.OUTCOME_COLUMNS},
        index=[f"c{i}" for i in range(40)],
    )
    b = a - 0.1  # a wins every column of every case
    v1 = backtest.verdict(a, b, a.index)
    v2 = backtest.verdict(a, b, a.index)
    assert v1 == v2
    assert v1["case_win"] == 1.0 and v1["lo"] == 1.0 and v1["hi"] == 1.0
    assert backtest.verdict(a, b, a.index[:3])["too_few"] is True


def test_quality_z_signs():
    p = pool(
        dep_role=["CB"] * 4,
        transfer_season=[2020] * 4,
        aerialDuelsWonPercentage=[60.0, 50.0, 40.0, 50.0],
        groundDuelsWonPercentage=[55.0, 50.0, 45.0, 50.0],
        dribbledPast=[0.2, 0.5, 0.8, 0.5],
        possessionLost=[5.0, 8.0, 11.0, 8.0],
    )
    z = backtest.quality_z(p)
    assert z.iloc[0] > z.iloc[1] > z.iloc[2]  # wins duels, keeps ball, rarely beaten -> best
