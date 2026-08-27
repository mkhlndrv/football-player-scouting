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

# Phase 0 Q2: Sofascore is the deeper work-rate source everywhere except Belgium and Denmark.
WORKRATE_PRIMARY = {comp: "sofascore" for comp in list(BIG5) + list(FEEDERS)}
WORKRATE_PRIMARY.update({"BE1": "fotmob", "DK1": "fotmob"})

TM_DUCKDB_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb"
)


def season_label(start: int) -> str:
    return f"{start}-{start + 1}"


def season_short(start: int) -> str:
    return f"{start % 100:02d}/{(start + 1) % 100:02d}"
