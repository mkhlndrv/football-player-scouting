import pandas as pd

from scout.models.fit import eligible_slots, role_shares, style_distance


def test_role_shares_and_eligibility():
    pm = pd.DataFrame(
        {
            "competition_id": ["GB1"] * 3,
            "season": [2023] * 3,
            "player_id": [1, 1, 1],
            "role": ["W", "ST", "CM"],
            "minutes": [1500, 400, 100],
        }
    )
    shares = role_shares(pm)
    assert shares.W.iloc[0] == 0.75 and shares.GK.iloc[0] == 0.0
    slots = eligible_slots(shares).iloc[0]
    assert slots.can_W and slots.can_ST and not slots.can_CM and not slots.can_GK


def test_style_distance_is_euclidean_and_nan_safe():
    a = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, float("nan")]})
    b = pd.DataFrame({"x": [3.0, 0.0], "y": [4.0, 2.0]})
    assert style_distance(a, b).tolist() == [5.0, 1.0]
