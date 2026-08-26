import subprocess

from common import OUT, TM_DUCKDB, TM_DUCKDB_URL, tm_connect, write_json

if not TM_DUCKDB.exists():
    subprocess.run(["curl", "-L", "-o", str(TM_DUCKDB), TM_DUCKDB_URL], check=True)

if not (OUT / "tm_schema.json").exists():
    con = tm_connect()
    schema = {}
    for (table,) in con.execute("SHOW TABLES").fetchall():
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        schema[table] = {"rows": n, "columns": {c[0]: c[1] for c in cols}}
    write_json("tm_schema.json", schema)
