import numpy as np
import pandas as pd

from scout.panel.stints import add_values
from scout.panel.values import value_at

VALUATIONS = pd.DataFrame(
    {
        "player_id": [1, 1, 1, 2],
        "date": ["2023-06-01", "2023-12-15", "2024-03-01", "2024-08-01"],
        "market_value_in_eur": [1e6, 2e6, 3e6, 5e5],
    }
)


def test_value_at_is_last_on_or_before_and_never_after():
    frame = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2],
            "when": ["2023-12-15", "2024-01-20", "2023-01-01", "2024-07-01"],
        }
    )
    out = value_at(frame, "when", VALUATIONS)
    assert out.value.tolist()[:2] == [2e6, 2e6]
    assert out.value_age_days.tolist()[:2] == [0, 36]
    assert np.isnan(out.value.iloc[2]) and np.isnan(out.value.iloc[3])


def test_add_values_keeps_lineup_only_rows_and_both_timestamps():
    stints = pd.DataFrame(
        {
            "tm_player_id": [1, 1],
            "club_id": [10, 20],
            "season": [2023, 2023],
            "minutes": [1500.0, np.nan],
            "first_date": ["2023-08-12", pd.NaT],
        }
    )
    out = add_values(stints, VALUATIONS)
    assert len(out) == 2 and np.isnan(out.minutes.iloc[1])
    assert out.value_at_start.tolist()[0] == 1e6 and out.value_july.tolist() == [1e6, 1e6]
    assert np.isnan(out.value_at_start.iloc[1])
    assert out.value_age_days_july.tolist() == [30, 30]
