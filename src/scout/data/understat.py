import json
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


MATCH_KINDS = ("player_match", "shots")


def _matches_without_rosters(data_dir: Path) -> set[int]:
    bad = set()
    for path in data_dir.glob("match_*.json"):
        rosters = json.loads(path.read_text()).get("rosters")
        if not isinstance(rosters, dict) or not rosters.get("h") or not rosters.get("a"):
            bad.add(int(path.stem.split("_")[1]))
    return bad


def _read_kind(reader, kind: str, log: Callable[[str], None]) -> pd.DataFrame:
    read = getattr(reader, _READER_METHOD[kind])
    try:
        return retry.until_done(lambda: read().reset_index(), log=log)
    except AttributeError:
        if kind not in MATCH_KINDS:
            raise
    # soccerdata assumes every match page carries rosters; an annulled match (Kiel-Bochum,
    # 2025-02-09, id 27930) keeps its fixture and xG but has none, so read by id without it
    skipped = _matches_without_rosters(Path(reader.data_dir))
    schedule = reader.read_schedule().reset_index()
    match_ids = [int(game) for game in schedule["game_id"] if int(game) not in skipped]
    log(f"understat {kind}: skipping matches without rosters {sorted(skipped)}")
    return retry.until_done(lambda: read(match_id=match_ids).reset_index(), log=log)


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
                frame = _read_kind(reader, kind, log)
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
