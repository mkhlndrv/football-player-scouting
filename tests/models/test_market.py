import numpy as np
import pandas as pd

from scout.models.market import features, fit, prepare

LEAGUES = ["GB1", "ES1"]


def _rows(n=400, seed=0):
    rng = np.random.default_rng(seed)
    point = rng.uniform(0, 1, n)
    age = rng.integers(19, 34, n)
    value_july = 10 ** (6 + point + rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "point": point,
            "history_point": np.where(rng.uniform(size=n) < 0.3, np.nan, point),
            "minutes": rng.integers(600, 3400, n),
            "age": age,
            "role": rng.choice(["ST", "W", "CM"], n),
            "competition_id": rng.choice(LEAGUES, n),
            "value_july": value_july,
            "value_next_july": value_july * 10 ** (0.5 * point + rng.normal(0, 0.05, n)),
            "club_elo_next": rng.uniform(1500, 1900, n),
        }
    )


def test_prepare_builds_target_and_fills_history():
    out = prepare(_rows())
    assert out.history_point.notna().all() and (out.y == np.log10(out.value_next_july)).all()
    assert features(out, LEAGUES).shape[1] == 8


def test_fit_is_monotone_in_contribution():
    prepared = prepare(_rows())
    model = fit(prepared, LEAGUES)
    base = features(prepared, LEAGUES)
    higher = base.assign(point=base.point + 0.1)
    assert (model.predict(higher) >= model.predict(base)).all()
