# Running the slow pulls on a server

The injury scrape (Transfermarkt pages, ~3 s each) and the ClubElo history pull (60–120 s per
answer) take days on a laptop that sleeps. Both resume from what is on disk, so they can run
anywhere and come back as files. Nothing else needs the server: Understat, Sofascore, FotMob,
the Transfermarkt DuckDB and reep are already complete locally (or download themselves).

## Server

AWS `eu-central-1`, Ubuntu 24.04, `t3.small`, 20 GB disk, security group: SSH from your IP only.

## Once, from the laptop

```bash
deploy/sync_to_server.sh ubuntu@HOST ~/.ssh/key.pem      # repo + resume state (no venv, no caches)
ssh -i ~/.ssh/key.pem ubuntu@HOST 'curl -LsSf https://astral.sh/uv/install.sh | sh'
ssh -i ~/.ssh/key.pem ubuntu@HOST 'cd football-player-scouting && ~/.local/bin/uv sync --locked'
```

## Start the pulls (detached; survive logout)

```bash
ssh -i ~/.ssh/key.pem ubuntu@HOST 'cd football-player-scouting && \
  nohup ~/.local/bin/uv run python -u -m scout fetch --only transfermarkt,reep,injuries > injuries.log 2>&1 < /dev/null & disown'
ssh -i ~/.ssh/key.pem ubuntu@HOST 'cd football-player-scouting && \
  nohup ~/.local/bin/uv run python -u -m scout fetch --only transfermarkt,reep,clubelo > clubelo.log 2>&1 < /dev/null & disown'
```

`< /dev/null & disown` matters: without it the detached process keeps the SSH session open.

```bash
```

`transfermarkt,reep` first so each process has the DuckDB and the reep files before it needs
them (both are no-ops once downloaded). Big-5 clubs and players go first in both queues.

## Check

```bash
ssh -i ~/.ssh/key.pem ubuntu@HOST 'cd football-player-scouting && tail -2 injuries.log clubelo.log'
```

A `403`/`429` line in `injuries.log` means Transfermarkt blocked the IP: stop, wait an hour,
restart (the fetcher never caches a block). `502`/timeouts in `clubelo.log` are normal.

## Bring the results back

```bash
deploy/sync_from_server.sh ubuntu@HOST ~/.ssh/key.pem
make data                                                # regenerates models/phase1_data_report.json
```

Then terminate the instance.
