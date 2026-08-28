from unittest.mock import patch

import pandas as pd
import requests

from scout.data import injuries

HEADER = (
    "<tr>"
    + "".join(
        f"<th>{h}</th>" for h in ["Season", "Injury", "from", "until", "Days", "Games missed"]
    )
    + "</tr>"
)


def _table(*rows):
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead>{HEADER}</thead><tbody>{body}</tbody></table>"


PAGE = _table(
    ["23/24", "Hamstring", "02/03/2024", "05/04/2024", "34 days", "6"],
    ["18/19", "Back injury", "04/07/2018", "06/07/2018", "3 days", "-"],
)
PAGE_2 = _table(["12/13", "Knock", "01/10/2012", "09/10/2012", "8 days", "1"])
NO_TABLE = "<html><p>no injuries</p></html>"


def _response(body):
    response = requests.Response()
    response.status_code = 200
    response._content = body.encode()
    return response


def test_injury_url_and_page_url():
    profile = "https://www.transfermarkt.co.uk/harry-kane/profil/spieler/132098"
    assert (
        injuries.page_url(profile, 1)
        == "https://www.transfermarkt.co.uk/harry-kane/verletzungen/spieler/132098"
    )
    assert (
        injuries.page_url(profile, 2)
        == "https://www.transfermarkt.co.uk/harry-kane/verletzungen/spieler/132098/page/2"
    )


def test_parse_page_types_and_missing():
    spells = injuries.parse_page(PAGE)
    assert list(spells.columns) == [
        "season",
        "injury",
        "from_date",
        "until_date",
        "days",
        "games_missed",
    ]
    assert spells.from_date.iloc[0] == pd.Timestamp("2024-03-02") and spells.days.iloc[0] == 34
    assert spells.games_missed.iloc[0] == 6 and pd.isna(spells.games_missed.iloc[1])  # '-' -> NaN
    assert injuries.parse_page(NO_TABLE).empty


def test_fetch_injuries_walks_pages_until_empty_or_repeated():
    bodies = {1: PAGE, 2: PAGE_2, 3: PAGE_2}  # page 3 repeats page 2: stop

    def fake_get(url, **kwargs):
        page = int(url.rsplit("/page/", 1)[1]) if "/page/" in url else 1
        return _response(bodies.get(page, NO_TABLE))

    with patch.object(injuries, "polite_get", fake_get):
        spells = injuries.fetch_injuries("https://www.transfermarkt.co.uk/x/profil/spieler/1")
    assert len(spells) == 3 and spells.season.tolist() == ["23/24", "18/19", "12/13"]


def test_pull_all_appends_and_marks_players_without_spells(tmp_path, monkeypatch):
    monkeypatch.setattr(injuries, "RAW", tmp_path)
    monkeypatch.setattr("scout.data.retry.time.sleep", lambda s: None)

    def fake_get(url, **kwargs):
        return _response(PAGE if "/spieler/1" in url and "/page/" not in url else NO_TABLE)

    players = pd.DataFrame(
        {
            "tm_player_id": [1, 2],
            "url": ["https://t/x/profil/spieler/1", "https://t/y/profil/spieler/2"],
        }
    )
    with patch.object(injuries, "polite_get", fake_get):
        injuries.pull_all(players, log=lambda m: None)
        injuries.pull_all(players, log=lambda m: None)  # second run: nothing new
    spells = injuries.load()
    assert set(spells.tm_player_id) == {1, 2}
    assert len(spells[spells.tm_player_id == 1]) == 2
    assert spells[spells.tm_player_id == 2].season.isna().all()  # fetched, no spells: marker row


def test_fetch_injuries_stops_at_pages_older_than_the_panel(monkeypatch):
    from scout.data import injuries

    def page(page_no):
        year = 2020 if page_no == 1 else 2010
        return (
            "<table><tr><th>Season</th><th>Injury</th><th>from</th><th>until</th><th>Days</th>"
            f"<th>Games missed</th></tr><tr><td>{year}/{year + 1}</td><td>Knock</td>"
            f"<td>01/09/{year}</td><td>10/09/{year}</td><td>9 days</td><td>1</td></tr></table>"
        )

    calls = []

    class Response:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    def fake_get(url, *args, **kwargs):
        calls.append(url)
        return Response(page(2 if "/page/" in url else 1))

    monkeypatch.setattr(injuries, "polite_get", fake_get)
    spells = injuries.fetch_injuries("https://www.transfermarkt.com/x/profil/spieler/1")
    assert len(calls) == 2  # page 2 already predates 2014-07-01, page 3 is never asked for
    assert len(spells) == 2
