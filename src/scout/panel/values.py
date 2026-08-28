import pandas as pd


def value_at(frame: pd.DataFrame, date_col: str, valuations: pd.DataFrame) -> pd.DataFrame:
    """Last Transfermarkt valuation on or before `frame[date_col]` per player, with its age in
    days (notebook 01, Part 1b). Never reads a later valuation; NaN when none exists."""
    keyed = frame[["player_id", date_col]].rename(columns={date_col: "date"})
    keyed = keyed.assign(date=pd.to_datetime(keyed["date"])).dropna(subset=["date"])
    keyed = keyed.reset_index().sort_values("date")
    history = valuations.rename(columns={"date": "valued_on"})
    history = history.assign(date=pd.to_datetime(history["valued_on"])).sort_values("date")
    hit = pd.merge_asof(keyed, history, on="date", by="player_id", direction="backward")
    hit = hit.set_index("index").reindex(frame.index)
    age = (hit["date"] - pd.to_datetime(hit["valued_on"])).dt.days
    return pd.DataFrame(
        {"value": hit["market_value_in_eur"], "value_age_days": age}, index=frame.index
    )
