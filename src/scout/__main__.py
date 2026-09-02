# determinism: HistGradientBoosting is not bit-reproducible across OpenMP threads
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from collections.abc import Callable

from scout import (
    config,
    report,
    train,
    train_confirmation,
    train_phase3,
    train_phase4,
    train_phase5,
    train_phase6,
)
from scout.data import clubelo, fotmob, injuries, reep, sofascore, understat
from scout.data import transfermarkt as tm

COMPS = list(config.BIG5) + list(config.FEEDERS)
SEASONS = list(config.SEASONS)


def _understat() -> None:
    understat.pull(list(config.BIG5.values()), SEASONS)


def _sofascore() -> None:
    for comp in COMPS:
        sofascore.pull_league(comp, SEASONS)


def _fotmob() -> None:
    for comp in COMPS:
        fotmob.pull_league(comp, SEASONS)


def _injuries() -> None:
    injuries.pull_all(injuries.panel_players(COMPS, SEASONS))


def _clubelo() -> None:
    clubs = clubelo.resolved_clubs(COMPS, SEASONS)
    clubelo.pull_histories(clubs["team_name"].unique().tolist())


# every pull resumes from what is on disk, so a rerun only fills the gaps
SOURCES: dict[str, Callable[[], None]] = {
    "transfermarkt": lambda: tm.ensure_duckdb(),
    "reep": lambda: reep.ensure_files(),
    "understat": _understat,
    "sofascore": _sofascore,
    "fotmob": _fotmob,
    "injuries": _injuries,
    "clubelo": _clubelo,
}


def fetch(only: list[str] | None = None) -> None:
    for name, run in SOURCES.items():
        if only is None or name in only:
            print(f"fetch: {name}", flush=True)
            run()


def main() -> None:
    parser = argparse.ArgumentParser(prog="scout")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch_cmd = sub.add_parser("fetch", help="pull raw inputs (resumable)")
    fetch_cmd.add_argument("--only", help=f"comma-separated subset of {', '.join(SOURCES)}")
    train_cmd = sub.add_parser("train", help="fit a model stage and write its artifacts")
    train_cmd.add_argument(
        "stage",
        choices=[
            "contribution",
            *train_phase3.STAGES,
            *train_phase4.STAGES,
            *train_phase5.STAGES,
            *train_confirmation.STAGES,
            *train_phase6.STAGES,
            "all",
        ],
    )
    data = sub.add_parser("data", help="fetch raw inputs, build the panel tables, write the report")
    data.add_argument("--skip-fetch", action="store_true", help="use what is on disk")
    args = parser.parse_args()
    if args.command == "fetch":
        only = args.only.split(",") if args.only else None
        unknown = set(only or []) - set(SOURCES)
        if unknown:
            parser.error(f"unknown sources: {sorted(unknown)}")
        fetch(only)
    elif args.command == "train":
        if args.stage in ("contribution", "all"):
            for name, path in train.run_contribution().items():
                print(f"{name}: {path}")
        for stage, run in {
            **train_phase3.STAGES,
            **train_phase4.STAGES,
            **train_phase5.STAGES,
            **train_confirmation.STAGES,
            **train_phase6.STAGES,
        }.items():
            if args.stage in (stage, "all"):
                print(f"{stage}: {run()}")
    elif args.command == "data":
        if not args.skip_fetch:
            fetch()
        print("report written to", report.write_report())


if __name__ == "__main__":
    main()
