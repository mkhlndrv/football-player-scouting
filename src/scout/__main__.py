import argparse

from scout import config, report
from scout.data import fotmob, injuries, reep, sofascore, understat
from scout.data import transfermarkt as tm


def fetch() -> None:
    """Every pull resumes from what is on disk, so a rerun only fills the gaps."""
    tm.ensure_duckdb()
    reep.ensure_files()
    comps = list(config.BIG5) + list(config.FEEDERS)
    seasons = list(config.SEASONS)
    understat.pull(list(config.BIG5.values()), seasons)
    for comp in comps:
        sofascore.pull_league(comp, seasons)
        fotmob.pull_league(comp, seasons)
    players = tm.load_table("players")
    panel_ids = tm.load_player_club_seasons(comps, seasons).tm_player_id.unique()
    players = players[players.player_id.isin(panel_ids)]
    injuries.pull_all(
        players.rename(columns={"player_id": "tm_player_id"})[["tm_player_id", "url"]]
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="scout")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data", help="fetch raw inputs, build the panel tables, write the report")
    data.add_argument("--skip-fetch", action="store_true", help="use what is on disk")
    args = parser.parse_args()
    if args.command == "data":
        if not args.skip_fetch:
            fetch()
        print("report written to", report.write_report())


if __name__ == "__main__":
    main()
