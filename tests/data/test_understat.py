import pandas as pd

from scout.data import understat


class FakeReader:
    def __init__(self, league, season, fail_once=None):
        self.league, self.season, self.fail_once = league, season, fail_once

    def _frame(self, kind):
        if self.fail_once is not None and self.fail_once.pop(kind, False):
            raise ConnectionError("dropped")
        code = f"{self.season % 100:02d}{(self.season + 1) % 100:02d}"
        frame = pd.DataFrame({"league": [self.league], "season": [code], "kind": [kind]})
        return frame.set_index(["league", "season"])

    def read_player_season_stats(self):
        return self._frame("player_season")

    def read_player_match_stats(self):
        return self._frame("player_match")

    def read_shot_events(self):
        return self._frame("shots")

    def read_team_match_stats(self):
        return self._frame("team_match")

    def read_schedule(self):
        return self._frame("schedule")


def test_pull_writes_each_kind_once_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(understat, "RAW", tmp_path)
    monkeypatch.setattr("scout.data.retry.time.sleep", lambda s: None)
    made = []

    def factory(league, season):
        made.append((league, season))
        return FakeReader(league, season, fail_once={"shots": True})

    understat.pull(["ENG-Premier League"], [2023], reader_factory=factory, log=lambda m: None)
    for kind in understat.KINDS:
        assert understat.raw_path(kind, "ENG-Premier League", 2023).exists()
    readers_made = len(made)
    understat.pull(["ENG-Premier League"], [2023], reader_factory=factory, log=lambda m: None)
    assert len(made) == readers_made  # nothing re-read once the files exist


def test_load_parses_season_start_year(tmp_path, monkeypatch):
    monkeypatch.setattr(understat, "RAW", tmp_path)
    understat.pull(["ENG-Premier League"], [2023], reader_factory=FakeReader, log=lambda m: None)
    shots = understat.load("shots")
    assert shots.season.tolist() == [2023] and shots.kind.tolist() == ["shots"]


def test_pull_skips_matches_without_rosters(tmp_path, monkeypatch):
    from scout.data import understat

    monkeypatch.setattr(understat, "RAW", tmp_path / "raw")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "match_1.json").write_text('{"rosters": {"h": {"1": {}}, "a": {"2": {}}}}')
    (cache / "match_2.json").write_text('{"rosters": {"h": [], "a": []}}')

    class Reader:
        data_dir = cache

        def read_schedule(self):
            return pd.DataFrame({"game_id": [1, 2]}).set_index("game_id")

        def read_player_match_stats(self, match_id=None):
            if match_id is None:
                raise AttributeError("'list' object has no attribute 'values'")
            return pd.DataFrame({"game_id": match_id, "minutes": [90] * len(match_id)})

    logged = []
    understat.pull(
        ["L"],
        [2024],
        kinds=("player_match",),
        reader_factory=lambda league, season: Reader(),
        log=logged.append,
    )
    out = pd.read_parquet(understat.raw_path("player_match", "L", 2024))
    assert out.game_id.tolist() == [1]
    assert any("skipping matches without rosters [2]" in line for line in logged)
