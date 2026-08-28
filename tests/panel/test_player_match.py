import pandas as pd
import pytest

from scout.panel.player_match import ROLE_OF_CODE, ROLES, assign_roles


def _rows(positions, minutes, player_id=1):
    n = len(positions)
    return pd.DataFrame(
        {
            "league": ["L"] * n,
            "season": [2023] * n,
            "team_id": [7] * n,
            "player_id": [player_id] * n,
            "position": positions,
            "minutes": minutes,
        }
    )


def test_every_code_maps_to_one_of_six_roles():
    assert set(ROLE_OF_CODE.values()) == set(ROLES) and len(ROLES) == 6
    assert ROLE_OF_CODE["AMC"] == "W" and ROLE_OF_CODE["DMC"] == "CM"
    assert ROLE_OF_CODE["DML"] == "FB"


def test_sub_inherits_main_role_by_starter_minutes():
    out = assign_roles(_rows(["MC", "DMC", "AMC", "Sub"], [90, 90, 900, 20]))
    assert out.role.tolist() == ["CM", "CM", "W", "W"]
    assert out.is_sub.tolist() == [False, False, False, True]


def test_sub_only_player_season_has_no_role():
    out = assign_roles(_rows(["Sub", "Sub"], [10, 15]))
    assert out.role.isna().all()


def test_side_is_the_code_suffix():
    out = assign_roles(_rows(["DL", "AMR", "DC", "Sub"], [90, 90, 90, 5]))
    assert out.side.tolist()[:2] == ["L", "R"]
    assert out.side.isna().tolist() == [False, False, True, True]


def test_unknown_code_raises():
    with pytest.raises(ValueError, match="unmapped"):
        assign_roles(_rows(["XYZ"], [90]))
