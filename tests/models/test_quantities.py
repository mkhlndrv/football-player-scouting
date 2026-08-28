import pandas as pd

from scout.models.quantities import (
    MIN_MINUTES,
    ROLE_QUANTITIES,
    penalty_xg,
    season_role_per90,
)


def test_role_table_covers_six_roles_without_goals():
    assert set(ROLE_QUANTITIES) == {"GK", "CB", "FB", "CM", "W", "ST"}
    for lists in ROLE_QUANTITIES.values():
        assert not {"goals", "assists"} & (set(lists["understat"]) | set(lists["workrate"]))
    assert "xa" not in ROLE_QUANTITIES["CB"]["understat"]
    assert MIN_MINUTES == 600


def test_penalty_xg_uses_the_understat_signature():
    shots = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "player_id": [7, 7, 8],
            "situation": [None, "OpenPlay", None],
            "xg": [0.7612, 0.7612, 0.05],
        }
    )
    assert penalty_xg(shots).to_dict() == {(1, 7): 0.7612}


def test_season_role_per90_subtracts_penalties_and_scales_to_90():
    pm = pd.DataFrame(
        {
            "competition_id": ["GB1"] * 2,
            "season": [2023] * 2,
            "player_id": [7, 7],
            "role": ["ST", "ST"],
            "game_id": [1, 2],
            "minutes": [90, 90],
            "xg": [1.7612, 0.5],
            "xa": [0.2, 0.0],
            "key_passes": [2, 1],
            "shots": [5, 2],
            "xg_chain": [2.0, 1.0],
            "xg_buildup": [0.5, 0.5],
            "goals": [2, 0],
            "assists": [0, 0],
        }
    )
    shots = pd.DataFrame({"game_id": [1], "player_id": [7], "situation": [None], "xg": [0.7612]})
    out = season_role_per90(pm, shots)
    assert len(out) == 1 and out.minutes.iloc[0] == 180
    assert abs(out.npxg.iloc[0] - (1.0 + 0.5) / 180 * 90) < 1e-9
    assert out.key_passes.iloc[0] == 1.5
