import pandas as pd

from scout.data import reep


def test_load_people_is_raw_strings(tmp_path):
    (tmp_path / "people.csv").write_text(
        "reep_id,name,key_transfermarkt,key_sofascore,key_understat,key_fotmob\n"
        "r1,Koke,74229,848057,,\n"
        "r2,Koke Vegas,299715,,,\n"
    )
    people = reep.load_people(tmp_path)
    assert people.key_transfermarkt.tolist() == ["74229", "299715"]
    assert people.key_sofascore.isna().sum() == 1


def test_transfermarkt_keys_maps_provider_to_tm_and_keeps_first_duplicate():
    people = pd.DataFrame(
        {
            "key_sofascore": ["1", "2", "3", "1"],
            "key_transfermarkt": ["10", None, "30", "99"],
        }
    )
    keys = reep.transfermarkt_keys(people, "sofascore")
    assert keys.to_dict() == {"1": "10", "3": "30"}
