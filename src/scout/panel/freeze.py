import pandas as pd

# spec 5.2: no feature uses data after the freeze date. A season counts as known once it has
# ended (30 June); rows with their own date are cut at the date itself.


class LeakageError(ValueError):
    pass


def season_end(season: int) -> pd.Timestamp:
    return pd.Timestamp(year=season + 1, month=6, day=30)


def seasons_before(seasons: list[int], as_of: pd.Timestamp) -> list[int]:
    return [season for season in seasons if season_end(season) <= as_of]


def refuse_after(seasons: list[int] | None, as_of: pd.Timestamp) -> None:
    """A caller asking for a season that ends after the freeze date is refused, not trimmed."""
    if seasons is None:
        return
    late = [season for season in seasons if season_end(season) > as_of]
    if late:
        raise LeakageError(f"seasons {late} end after the freeze date {as_of.date()}")


def cut(
    frame: pd.DataFrame,
    as_of: pd.Timestamp,
    date_col: str | None = None,
    season_col: str = "season",
) -> pd.DataFrame:
    if date_col is not None:
        return frame[pd.to_datetime(frame[date_col]) <= as_of]
    return frame[frame[season_col].map(season_end) <= as_of]


def assert_frozen(
    frame: pd.DataFrame,
    as_of: pd.Timestamp,
    date_col: str | None = None,
    season_col: str = "season",
) -> None:
    leaked = len(frame) - len(cut(frame, as_of, date_col, season_col))
    if leaked:
        raise LeakageError(f"{leaked} rows fall after the freeze date {as_of.date()}")
