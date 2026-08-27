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
