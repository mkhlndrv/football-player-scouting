import numpy as np
import pandas as pd
import pytest

from scout.models.similarity import FEATURES, neighbours, shot_locations


def test_shot_locations_need_ten_shots_and_measure_the_box():
    shots = pd.DataFrame(
        {
            "competition_id": ["GB1"] * 12,
            "season": [2023] * 12,
            "player_id": [1] * 11 + [2],
            "result": ["Goal"] * 12,
            "location_x": [0.9] * 11 + [0.5],
            "location_y": [0.5] * 11 + [0.5],
        }
    )
    out = shot_locations(shots).set_index("player_id")
    assert out.loc[1, "box_share"] == 1.0 and out.loc[1, "shot_dist"] == pytest.approx(10.5)
    assert np.isnan(out.loc[2, "shot_dist"])


def test_neighbours_are_within_role_and_season_and_exclude_self():
    rng = np.random.default_rng(0)
    rows = pd.DataFrame(rng.normal(size=(30, len(FEATURES))), columns=FEATURES)
    rows["role"] = ["W"] * 15 + ["ST"] * 15
    rows["season"] = 2023
    rows["player_id"] = range(30)
    out = neighbours(rows, k=3).set_index("player_id")
    assert len(out) == 30 and all(len(n) == 3 for n in out.neighbours)
    assert all(pid not in out.loc[pid, "neighbours"] for pid in out.index)
    assert all(n < 15 for n in out.loc[0, "neighbours"])  # a winger's neighbours are wingers
