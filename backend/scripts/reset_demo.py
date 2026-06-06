from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtime"


def main() -> None:
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    print(f"Reset DataMeta runtime at {RUNTIME}")


if __name__ == "__main__":
    main()
