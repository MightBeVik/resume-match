"""Run the full jobs pipeline end to end."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    steps = [
        "01_ingest.py",
        "02_transform.py",
        "03_enrich.py",
        "04_synthesize.py",
        "05_load.py",
        "06_validate.py",
    ]

    for step in steps:
        print(f"[run_all] Running {step}")
        subprocess.run([sys.executable, str(ROOT / step)], check=True)

    print("[run_all] Pipeline completed successfully")


if __name__ == "__main__":
    main()