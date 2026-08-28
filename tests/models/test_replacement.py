import pandas as pd
import pytest

from scout.models.replacement import PERCENTILE, replacement_level, surplus


def test_replacement_level_is_the_low_percentile_per_role_season():
    regulars = pd.DataFrame(
        {
            "role": ["ST"] * 5 + ["W"] * 5,
            "season": [2023] * 10,
            "raw": list(range(1, 6)) + list(range(10, 15)),
        }
    )
    levels = replacement_level(regulars, "raw").set_index("role")
    assert PERCENTILE == 0.20
    assert levels.loc["ST", "replacement_level"] == pytest.approx(regulars.raw[:5].quantile(0.2))
    assert levels.loc["W", "players"] == 5


def test_surplus_subtracts_the_role_season_level():
    regulars = pd.DataFrame(
        {"role": ["ST"] * 5, "season": [2023] * 5, "raw": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    levels = replacement_level(regulars, "raw")
    out = surplus(regulars, "raw", levels)
    assert out.iloc[-1] == pytest.approx(5.0 - regulars.raw.quantile(0.2))
    assert out.index.equals(regulars.index)
