"""Read-only DM serial-to-CAN preflight on Windows, Linux and macOS."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.rs_hardware_check import main

if __name__ == "__main__":
    raise SystemExit(main("rebotarm_dm.yaml", "logs/dm_preflight.csv"))
