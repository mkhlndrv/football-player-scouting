import pandas as pd
import pytest

from scout.models.resale import empirical_bands, interval


def test_empirical_bands_and_interval():
    residuals = {
        1: pd.Series([-0.2, -0.1, 0.0, 0.1, 0.2, float("nan")]),
        2: pd.Series([-0.4, 0.0, 0.4]),
    }
    bands = empirical_bands(residuals)
    assert bands[1][0] == pytest.approx(-0.16) and bands[1][1] == pytest.approx(0.16)
    band = interval(pd.Series([6.0, 7.0]), bands[2])
    assert band.lo.tolist() == pytest.approx([6.0 + bands[2][0], 7.0 + bands[2][0]])
    assert (band.hi > band.lo).all()
