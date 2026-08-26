import pandas as pd

from scout.identity import last_token, match_players, normalize_name


def test_normalize_name_strips_accents_case_and_punctuation():
    assert normalize_name("Kylian Mbappé") == "kylian mbappe"
    assert normalize_name("N'Golo  Kanté") == "ngolo kante"
    assert normalize_name("Bertuğ Yıldırım") == "bertug yildirim"


def test_last_token():
    assert last_token("Robert Lewandowski") == "lewandowski"


def test_match_players_cascade():
    left = pd.DataFrame(
        {
            "name": ["Kylian Mbappé", "R. Lewandowski", "Unknown Person", "João Félix"],
            "club_key": ["real", "barca", "real", "barca"],
            "season": [2024, 2024, 2024, 2024],
        }
    )
    right = pd.DataFrame(
        {
            "right_id": [1, 2, 3, 4],
            "name": ["Kylian Mbappe", "Robert Lewandowski", "Someone Else", "Joao Felix"],
            "club_key": ["real", "barca", "real", "barca"],
            "season": [2024, 2024, 2024, 2024],
        }
    )
    out = match_players(left, right)
    assert out.loc[0, "right_id"] == 1 and out.loc[0, "method"] == "exact"
    assert out.loc[1, "right_id"] == 2 and out.loc[1, "method"] == "last_token"
    assert pd.isna(out.loc[2, "right_id"]) and out.loc[2, "method"] == "unmatched"
    assert out.loc[3, "right_id"] == 4 and out.loc[3, "method"] == "exact"


def test_match_players_last_token_requires_uniqueness():
    left = pd.DataFrame({"name": ["J. Silva"], "club_key": ["x"], "season": [2020]})
    right = pd.DataFrame(
        {
            "right_id": [1, 2],
            "name": ["Joao Silva", "Jose Silva"],
            "club_key": ["x", "x"],
            "season": [2020, 2020],
        }
    )
    out = match_players(left, right)
    assert out.loc[0, "method"] == "unmatched"
