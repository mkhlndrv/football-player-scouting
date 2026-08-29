import pandas as pd
import pytest

from scout.models.trajectory import horizon_inflation, project, role_curve


def test_role_curve_and_projection_walk_the_age_bands():
    pairs = pd.DataFrame(
        {
            "role": ["W"] * 4,
            "age": [21, 22, 31, 32],
            "point": [0.3, 0.3, 0.4, 0.4],
            "point_next": [0.32, 0.32, 0.37, 0.37],
        }
    )
    curve = role_curve(pairs)
    assert curve[("W", "21-22")] == pytest.approx(0.02) and curve[("W", "31-32")] == pytest.approx(
        -0.03
    )
    point, age, role = pd.Series([0.3, 0.4]), pd.Series([21, 31]), pd.Series(["W", "W"])
    one = project(point, age, role, curve, 1)
    assert one.tolist() == pytest.approx([0.32, 0.37])
    two = project(point, age, role, curve, 2)
    assert two.iloc[0] == pytest.approx(0.34)  # 21 -> 22 stays in the 21-22 band
    assert two.iloc[1] == pytest.approx(0.34)  # 31 -> 32 stays in the 31-32 band


def test_horizon_inflation_is_one_for_calibrated_z():
    import numpy as np

    z = pd.Series(np.random.default_rng(0).normal(0, 1, 20000))
    assert horizon_inflation(z) == pytest.approx(1.0, abs=0.03)
