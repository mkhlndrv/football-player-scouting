from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"

# Transfermarkt competition id -> soccerdata/Understat league name (Tier 1, spec §2)
BIG5 = {
    "GB1": "ENG-Premier League",
    "ES1": "ESP-La Liga",
    "IT1": "ITA-Serie A",
    "L1": "GER-Bundesliga",
    "FR1": "FRA-Ligue 1",
}
# Transfermarkt competition id -> country (Tier 2, Phase 0 Q3)
FEEDERS = {
    "BE1": "Belgium",
    "NL1": "Netherlands",
    "PO1": "Portugal",
    "TR1": "Turkey",
    "C1": "Switzerland",
    "BRA1": "Brazil",
    "A1": "Austria",
    "DK1": "Denmark",
}
SEASONS = list(range(2014, 2026))  # start years, 2014-15 .. 2025-26 (Understat's range)

# Sofascore unique-tournament ids, verified by name in notebooks/01 Part 5 (2026-08-27).
SOFASCORE_TOURNAMENTS = {
    "GB1": 17,
    "ES1": 8,
    "IT1": 23,
    "L1": 35,
    "FR1": 34,
    "BE1": 38,
    "NL1": 37,
    "PO1": 238,
    "TR1": 52,
    "C1": 215,
    "BRA1": 325,
    "A1": 45,
    "DK1": 39,
}

# FotMob league ids, verified by name in notebooks/01 Part 6d (2026-08-27). Stats are pulled per
# stat from data.fotmob.com/stats/{league}/season/{season_id}/{stat}.json.
FOTMOB_LEAGUES = {
    "GB1": 47,
    "ES1": 87,
    "IT1": 55,
    "L1": 54,
    "FR1": 53,
    "BE1": 40,
    "NL1": 57,
    "PO1": 61,
    "TR1": 71,
    "C1": 69,
    "BRA1": 268,
    "A1": 38,
    "DK1": 46,
}

# Phase 0 Q2: Sofascore is the deeper work-rate source everywhere except Belgium and Denmark.
WORKRATE_PRIMARY = {comp: "sofascore" for comp in list(BIG5) + list(FEEDERS)}
WORKRATE_PRIMARY.update({"BE1": "fotmob", "DK1": "fotmob"})

TM_DUCKDB_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb"
)
REEP_URL = "https://raw.githubusercontent.com/withqwerty/reep/main/data"


def season_label(start: int) -> str:
    return f"{start}-{start + 1}"


def season_short(start: int) -> str:
    return f"{start % 100:02d}/{(start + 1) % 100:02d}"
