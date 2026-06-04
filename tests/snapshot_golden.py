"""Capture the current pipeline output as a 'golden' baseline.

Run once after the build is frozen and again whenever you intentionally
re-baseline.  Writes to tests/golden/:

    tests/golden/csvs/*.csv            - copy of every results/*.csv
    tests/golden/mask_hashes.txt       - sha256 of every <fit_folder>/*_fit_mask.npy
    tests/golden/MANIFEST.txt          - timestamp + counts for sanity

Mask files live on the data drive (default E:\\18_RK_Circadian\\data\\raw on
Windows) and number in the thousands, so we store sha256 hashes rather than
copying.  Hashes are deterministic per binary file and tiny.

Run from project root:
    uv run python tests/snapshot_golden.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

# -- paths --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
GOLDEN_DIR   = PROJECT_ROOT / "tests" / "golden"
GOLDEN_CSVS  = GOLDEN_DIR / "csvs"
HASH_FILE    = GOLDEN_DIR / "mask_hashes.txt"
MANIFEST     = GOLDEN_DIR / "MANIFEST.txt"

# Cross-OS data roots; first match wins
DATA_ROOTS = [
    Path(r"E:\18_RK_Circadian\data\raw"),
    Path("/media/mint/BRPresbkup/18_RK_Circadian/data/raw"),
]


def _pick_data_root() -> Path | None:
    for root in DATA_ROOTS:
        if root.exists():
            return root
    return None


def _sha256(p: Path, chunk: int = 1 << 20) -> str:
    """ Calculate sha256 hash of file at p.
        A bit more complicated than the default idiom,
        because it loads the file into memory chunkwise.
    """

    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def snapshot_csvs() -> int:
    GOLDEN_CSVS.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(RESULTS_DIR.glob("*.csv")):
        dst = GOLDEN_CSVS / src.name
        shutil.copy2(src, dst)
        n += 1
        print(f"  CSV  {src.name}  ({src.stat().st_size:,} bytes)")
    return n


def snapshot_mask_hashes(data_root: Path) -> int:
    n = 0
    with open(HASH_FILE, "w", encoding="utf-8") as out:
        out.write(f"# sha256 of every *_fit_mask.npy under {data_root}\n")
        out.write(f"# captured {datetime.now().isoformat()}\n")
        for p in sorted(data_root.rglob("*_fit_mask.npy")):
            rel = p.relative_to(data_root).as_posix()
            digest = _sha256(p)
            out.write(f"{digest}  {rel}\n")
            n += 1
            if n % 200 == 0:
                print(f"  hashed {n} masks...")
    return n


def main() -> int:
    if not RESULTS_DIR.exists():
        print(f"ERROR: {RESULTS_DIR} not found", file=sys.stderr)
        return 1
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Snapshotting CSVs from {RESULTS_DIR}")
    n_csv = snapshot_csvs()

    data_root = _pick_data_root()
    if data_root is None:
        print(f"WARNING: no data root in {DATA_ROOTS} found; skipping mask hashes",
              file=sys.stderr)
        n_mask = 0
    else:
        print(f"Hashing *_fit_mask.npy under {data_root}")
        n_mask = snapshot_mask_hashes(data_root)

    with open(MANIFEST, "w", encoding="utf-8") as fh:
        fh.write(f"timestamp:   {datetime.now().isoformat()}\n")
        fh.write(f"csv_count:   {n_csv}\n")
        fh.write(f"mask_count:  {n_mask}\n")
        fh.write(f"data_root:   {data_root}\n")
        fh.write(f"results_dir: {RESULTS_DIR}\n")

    print(f"\nGolden captured: {n_csv} CSVs, {n_mask} mask hashes")
    print(f"  -> {GOLDEN_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
