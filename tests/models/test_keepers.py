import pandas as pd
import pytest

from scout.models.keepers import prevented_per90

KEYS = {"competition_id": "GB1", "season": 2023, "game_id": 1}


def test_prevented_per90_counts_the_opponents_on_target_shots_only():
    player_match = pd.DataFrame(
        [
            {**KEYS, "team_id": 10, "player_id": 7, "role": "GK", "minutes": 90},
            {**KEYS, "team_id": 20, "player_id": 8, "role": "GK", "minutes": 90},
            {**KEYS, "team_id": 10, "player_id": 9, "role": "CB", "minutes": 90},
        ]
    )
    shots = pd.DataFrame(
        [
            {**KEYS, "team_id": 20, "result": "Goal", "xg": 0.5},
            {**KEYS, "team_id": 20, "result": "Saved Shot", "xg": 0.3},
            {**KEYS, "team_id": 20, "result": "Missed Shot", "xg": 0.9},
            {**KEYS, "team_id": 10, "result": "Saved Shot", "xg": 0.2},
        ]
    )
    team_game = pd.DataFrame([{**KEYS, "team_id": 10}, {**KEYS, "team_id": 20}])
    out = prevented_per90(player_match, shots, team_game).set_index("player_id")
    assert out.loc[7, "xg_on_target"] == pytest.approx(0.8) and out.loc[7, "goals_conceded"] == 1
    assert out.loc[7, "prevented_per90"] == pytest.approx(-0.2)
    assert out.loc[8, "xg_on_target"] == pytest.approx(0.2) and out.loc[8, "goals_conceded"] == 0
    assert 9 not in out.index
