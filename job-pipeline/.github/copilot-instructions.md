Keep raw downloads in data/raw and never overwrite them in-place.
Use requests for HTTP, pandas for tabular transforms, sqlite3 for database writes.
All scripts must remain runnable standalone via python scripts/<name>.py.
Prefer pathlib.Path over os.path.
Print readable progress messages with [step] prefixes.
Raise explicit errors when downloads fail or joins produce zero rows.