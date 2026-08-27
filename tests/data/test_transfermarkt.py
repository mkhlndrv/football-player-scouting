import duckdb
import pandas as pd

from scout.data import transfermarkt as tm


def _tiny_db(path):
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE players(player_id INT, name VARCHAR, date_of_birth DATE, position VARCHAR,"
        " sub_position VARCHAR, url VARCHAR)"
    )
    con.execute(
        "INSERT INTO players VALUES (1,'A One','1995-01-01','Attack','Centre-Forward','u1'),"
        "(2,'B Two','1990-05-05','Goalkeeper','Goalkeeper','u2')"
    )
    con.execute(
        "CREATE TABLE clubs(club_id VARCHAR, name VARCHAR, domestic_competition_id VARCHAR,"
        " total_market_value DOUBLE)"
    )
    con.execute(
        "INSERT INTO clubs VALUES ('10','Arsenal FC','GB1',1e9),('11','Chelsea FC','GB1',9e8)"
    )
    con.execute(
        "CREATE TABLE games(game_id VARCHAR, competition_id VARCHAR, season VARCHAR, date DATE,"
        " home_club_id INT, away_club_id INT)"
    )
    con.execute(
        "INSERT INTO games VALUES ('g1','GB1','2023','2023-08-12',10,11),"
        "('g2','GB1','2023','2023-08-19',11,10)"
    )
    con.execute(
        "CREATE TABLE appearances(appearance_id VARCHAR, game_id INT, player_id INT,"
        " player_club_id VARCHAR, minutes_played INT)"
    )
    con.execute("INSERT INTO appearances VALUES ('a1',1,1,'10',90)")
    con.execute("UPDATE appearances SET game_id = 1")
    con.execute("ALTER TABLE appearances ALTER game_id TYPE VARCHAR")
    con.execute("UPDATE appearances SET game_id = 'g1'")
    con.execute(
        "CREATE TABLE game_lineups(game_lineups_id VARCHAR, game_id VARCHAR, player_id INT,"
        " club_id INT, type VARCHAR, position VARCHAR)"
    )
    con.execute(
        "INSERT INTO game_lineups VALUES ('l1','g1',1,10,'starting_lineup','Centre-Forward'),"
        "('l2','g1',2,10,'substitutes','Goalkeeper')"
    )
    con.close()
    return path


def test_player_club_seasons_unions_appearances_and_lineups(tmp_path):
    con = duckdb.connect(str(_tiny_db(tmp_path / "t.duckdb")), read_only=True)
    out = tm.load_player_club_seasons(["GB1"], [2023], con=con)
    assert set(out.tm_player_id) == {1, 2}
    starter = out[out.tm_player_id == 1].iloc[0]
    bench_only = out[out.tm_player_id == 2].iloc[0]
    assert starter.minutes == 90 and starter.source == "appearances"
    assert starter.club_id == 10 and starter.season == 2023 and starter.club_name == "Arsenal FC"
    assert pd.isna(bench_only.minutes) and bench_only.source == "lineups"  # NaN, never 0


def test_load_table_returns_raw(tmp_path):
    con = duckdb.connect(str(_tiny_db(tmp_path / "t.duckdb")), read_only=True)
    clubs = tm.load_table("clubs", con=con)
    assert list(clubs.columns) == [
        "club_id",
        "name",
        "domestic_competition_id",
        "total_market_value",
    ]
