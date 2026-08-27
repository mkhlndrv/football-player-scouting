from collections.abc import Callable
from pathlib import Path

import pandas as pd

from scout import config
from scout.data import retry

RAW = config.RAW
KINDS = ("player_season", "player_match", "shots", "team_match", "schedule")
_READER_METHOD = {
    "player_season": "read_player_season_stats",
    "player_match": "read_player_match_stats",
    "shots": "read_shot_events",
    "team_match": "read_team_match_stats",
    "schedule": "read_schedule",
}


def raw_path(kind: str, league: str, season: int) -> Path:
    return RAW / "understat" / kind / f"{league}_{season}.parquet"


def _soccerdata_reader(league: str, season: int):
    import soccerdata as sd

    return sd.Understat(
        leagues=[league],
        seasons=[config.season_label(season)],
        data_dir=config.CACHE / "understat",
    )


def pull(
    leagues: list[str],
    seasons: list[int],
    kinds: tuple[str, ...] = KINDS,
    reader_factory: Callable | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """One parquet per (kind, league, season); existing files are skipped, so a killed pull
    resumes where it stopped."""
    factory = reader_factory or _soccerdata_reader
    for league in leagues:
        for season in seasons:
            todo = [kind for kind in kinds if not raw_path(kind, league, season).exists()]
            if not todo:
                continue
            reader = factory(league, season)
            for kind in todo:
                read = getattr(reader, _READER_METHOD[kind])
                frame = retry.until_done(lambda read=read: read().reset_index(), log=log)
                frame.columns = [str(column).lower() for column in frame.columns]
                path = raw_path(kind, league, season)
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)
                log(f"understat {kind} {league} {season}: {len(frame)} rows")


def load(
    kind: str, leagues: list[str] | None = None, seasons: list[int] | None = None
) -> pd.DataFrame:
    frames = []
    for path in sorted((RAW / "understat" / kind).glob("*.parquet")):
        league, season = path.stem.rsplit("_", 1)
        if (leagues and league not in leagues) or (seasons and int(season) not in seasons):
            continue
        frames.append(pd.read_parquet(path))
    frame = pd.concat(frames, ignore_index=True)
    frame["season"] = frame["season"].astype(str).str[:2].map(lambda code: 2000 + int(code))
    return frame
