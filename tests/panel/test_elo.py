import numpy as np
import pandas as pd

from scout.panel.elo import elo_on_dates


def test_elo_on_dates_uses_the_interval_containing_the_date_and_never_reads_ahead():
    history = pd.DataFrame(
        {
            "From": pd.to_datetime(["2020-07-01", "2020-08-15"]),
            "To": pd.to_datetime(["2020-08-14", "2020-09-30"]),
            "Elo": [1700.0, 1720.0],
        }
    )
    dates = pd.Series(["2020-06-01", "2020-07-20", "2020-08-15", "2020-10-05"])
    out = elo_on_dates(history, dates)
    assert np.isnan(out[0]) and out[1] == 1700.0 and out[2] == 1720.0 and np.isnan(out[3])
    assert np.isnan(elo_on_dates(history.iloc[0:0], dates)).all()
