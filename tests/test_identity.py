import pandas as pd

from scout.identity import (
    build_team_lineage,
    last_token,
    load_overrides,
    match_players,
    normalize_name,
    resolve_player_ids,
)


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


def test_team_lineage_prefers_shortest_on_ties_and_applies_overrides():
    tm_clubs = pd.DataFrame(
        {
            "club_id": [131, 714, 621],
            "club_name": ["FC Barcelona", "RCD Espanyol Barcelona", "Athletic Bilbao"],
            "competition_id": ["ES1"] * 3,
        }
    )
    provider_teams = {
        "understat": pd.DataFrame(
            {"competition_id": ["ES1"] * 3, "team_name": ["Barcelona", "Espanyol", "Athletic Club"]}
        )
    }
    overrides = pd.DataFrame(
        {
            "provider": ["understat"],
            "competition_id": ["ES1"],
            "team_name": ["Athletic Club"],
            "club_id": [621],
        }
    )
    lineage = build_team_lineage(tm_clubs, provider_teams, overrides).set_index("team_name")
    assert lineage.loc["Barcelona", "club_id"] == 131  # not Espanyol: shortest tie wins
    assert lineage.loc["Espanyol", "club_id"] == 714
    assert lineage.loc["Athletic Club", "club_id"] == 621
    assert lineage.loc["Athletic Club", "source"] == "override"


def test_team_lineage_leaves_low_scores_unresolved_and_keeps_provider_ids():
    tm_clubs = pd.DataFrame(
        {"club_id": [3911], "club_name": ["Stade Brestois 29"], "competition_id": ["FR1"]}
    )
    provider_teams = {
        "sofascore": pd.DataFrame(
            {"competition_id": ["FR1"], "team_name": ["Brest"], "provider_team_id": [1715]}
        )
    }
    lineage = build_team_lineage(
        tm_clubs,
        provider_teams,
        pd.DataFrame(columns=["provider", "competition_id", "team_name", "club_id"]),
    )
    row = lineage.iloc[0]
    assert pd.isna(row.club_id) and row.source == "auto" and row.provider_team_id == 1715


def test_load_overrides_reads_committed_file():
    teams = load_overrides("teams")
    assert list(teams.columns) == ["provider", "competition_id", "team_name", "club_id"]
    assert (teams.provider == "understat").sum() == 11


def test_name_unique_ignores_mononyms():
    left = pd.DataFrame(
        {"name": ["Rafinha", "Bruno Henrique"], "club_key": ["a", "a"], "season": [2024, 2024]}
    )
    right = pd.DataFrame(
        {
            "right_id": [1, 2],
            "name": ["Rafinha", "Bruno Henrique"],
            "club_key": ["b", "b"],
            "season": [2024, 2024],
        }
    )
    out = match_players(left, right)
    assert out.loc[0, "method"] == "unmatched"
    assert out.loc[1, "right_id"] == 2 and out.loc[1, "method"] == "name_unique"


def test_default_fuzzy_threshold_keeps_spelling_variants():
    left = pd.DataFrame({"name": ["Yegor Yarmoliuk"], "club_key": ["x"], "season": [2023]})
    right = pd.DataFrame(
        {"right_id": [1], "name": ["Yehor Yarmolyuk"], "club_key": ["x"], "season": [2023]}
    )
    out = match_players(left, right)
    assert out.loc[0, "right_id"] == 1 and out.loc[0, "method"] == "fuzzy"
    assert match_players(left, right, min_fuzzy=92).loc[0, "method"] == "unmatched"


def test_resolve_player_ids_prefers_reep_then_minutes_weighted_mode():
    matches = pd.DataFrame(
        {
            "provider_id": [1, 1, 2, 2, 2, 3, 4],
            "right_id": [10, 11, 20, 21, 21, 30, pd.NA],
            "minutes": [3000, 100, 900, 500, 500, float("nan"), float("nan")],
        }
    )
    reep_keys = pd.Series({"1": "99"})
    out = resolve_player_ids(matches, reep_keys).set_index("provider_id")
    assert out.loc["1", "tm_player_id"] == "99" and out.loc["1", "source"] == "reep"
    assert out.loc["2", "tm_player_id"] == "21" and out.loc["2", "source"] == "cascade"
    assert out.loc["3", "tm_player_id"] == "30"
    assert pd.isna(out.loc["4", "tm_player_id"]) and out.loc["4", "source"] == "unmatched"
