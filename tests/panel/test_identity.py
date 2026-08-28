import pandas as pd

from scout.panel.identity import minutes_rate, resolve_provider, transfermarkt_side


def test_resolve_provider_uses_reep_then_cascade_and_reports_minutes_rate():
    tm_panel = pd.DataFrame(
        {
            "tm_player_id": [100, 200, 300],
            "name": ["Kylian Mbappe", "Vinicius Junior", "Someone Else"],
            "club_id": [418, 418, 418],
            "season": [2024, 2024, 2024],
            "competition_id": ["ES1"] * 3,
        }
    )
    rows = pd.DataFrame(
        {
            "competition_id": ["ES1"] * 3,
            "season": [2024] * 3,
            "sofascore_player_id": [1, 2, 3],
            "player_name": ["Kylian Mbappé", "Vini Jr.", "Nobody Known"],
            "team_name": ["Real Madrid"] * 3,
            "minutesPlayed": [3000.0, 2000.0, 1000.0],
        }
    )
    lineage = pd.DataFrame(
        {
            "provider": ["sofascore"],
            "competition_id": ["ES1"],
            "team_name": ["Real Madrid"],
            "club_id": [418],
        }
    )
    people = pd.DataFrame({"key_sofascore": ["2"], "key_transfermarkt": ["200"]})
    resolved = resolve_provider("sofascore", rows, transfermarkt_side(tm_panel), lineage, people)
    by_id = resolved.set_index("provider_id")
    assert by_id.loc["1", "tm_player_id"] == "100" and by_id.loc["1", "source"] == "cascade"
    assert by_id.loc["2", "tm_player_id"] == "200" and by_id.loc["2", "source"] == "reep"
    assert pd.isna(by_id.loc["3", "tm_player_id"])
    assert minutes_rate(resolved).loc["ES1", 2024] == 5000 / 6000
