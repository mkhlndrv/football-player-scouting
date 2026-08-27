from scout import config


def test_season_labels():
    assert config.season_label(2014) == "2014-2015"
    assert config.season_short(2014) == "14/15"
    assert config.SEASONS[0] == 2014 and config.SEASONS[-1] == 2025


def test_league_tables_are_consistent():
    assert set(config.BIG5) == {"GB1", "ES1", "IT1", "L1", "FR1"}
    assert set(config.FEEDERS) == {"BE1", "NL1", "PO1", "TR1", "C1", "BRA1", "A1", "DK1"}
    assert set(config.WORKRATE_PRIMARY) == set(config.BIG5) | set(config.FEEDERS)
    assert config.WORKRATE_PRIMARY["BE1"] == "fotmob"
    assert config.WORKRATE_PRIMARY["GB1"] == "sofascore"
