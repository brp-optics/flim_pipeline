"""Compare current pipeline outputs against the captured golden baseline.

Usage (from project root):
    uv run python tests/compare_outputs.py
    uv run python tests/compare_outputs.py --no-masks   # CSV+snapshots only
    uv run python tests/compare_outputs.py --quick      # CSVs only

Exit code is 0 iff every file matches.  Reports each mismatched file and the
first few differing rows (for CSVs) so regressions land in plain English.

Looks at three buckets:
  1. results/*.csv         vs  tests/golden/csvs/*.csv
  2. fit folder *_fit_mask.npy hashes  vs  tests/golden/mask_hashes.txt
  3. tests/snapshots/*.pkl vs  (none yet -- snapshots are compared run-to-run
                                 via assert_frame_equal once both exist)

When snapshots/golden_snapshots/ exists we also diff tests/snapshots/ against
it (per-pickle DataFrame compare).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT      = Path(__file__).resolve().parent.parent
RESULTS_DIR       = PROJECT_ROOT / "results"
GOLDEN_DIR        = PROJECT_ROOT / "tests" / "golden"
GOLDEN_CSVS       = GOLDEN_DIR / "csvs"
GOLDEN_HASHES     = GOLDEN_DIR / "mask_hashes.txt"
GOLDEN_SNAPSHOTS  = GOLDEN_DIR / "snapshots"
CURRENT_SNAPSHOTS = PROJECT_ROOT / "tests" / "snapshots"

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
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def compare_csvs() -> list[str]:
    """Return a list of failure messages; empty list means everything matched."""
    failures: list[str] = []
    if not GOLDEN_CSVS.exists():
        return [f"missing golden CSV directory: {GOLDEN_CSVS}"]
    golden_files = sorted(GOLDEN_CSVS.glob("*.csv"))
    current_files = {p.name for p in RESULTS_DIR.glob("*.csv")}
    for golden in golden_files:
        cur = RESULTS_DIR / golden.name
        if not cur.exists():
            failures.append(f"CSV missing in results/: {golden.name}")
            continue
        try:
            g = pd.read_csv(golden)
            c = pd.read_csv(cur)
            pd.testing.assert_frame_equal(c, g, check_dtype=False, atol=1e-8)
        except AssertionError as e:
            failures.append(f"CSV diff: {golden.name}\n    {str(e).splitlines()[0]}")
        except Exception as e:
            failures.append(f"CSV read/compare error: {golden.name}: {e}")
    # CSVs present in current but not golden are noted, not failed
    extras = sorted(current_files - {g.name for g in golden_files})
    if extras:
        print(f"  note: {len(extras)} CSV(s) in results/ but not in golden: "
              f"{', '.join(extras)}")
    return failures


def compare_mask_hashes() -> list[str]:
    failures: list[str] = []
    if not GOLDEN_HASHES.exists():
        return [f"missing golden hash file: {GOLDEN_HASHES}"]
    data_root = _pick_data_root()
    if data_root is None:
        return [f"no data root found in {DATA_ROOTS}"]
    expected: dict[str, str] = {}
    with open(GOLDEN_HASHES, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, rel = line.split(None, 1)
            expected[rel] = digest
    n_checked = 0
    n_missing = 0
    n_mismatch = 0
    for rel, exp_digest in expected.items():
        p = data_root / rel
        if not p.exists():
            failures.append(f"mask missing: {rel}")
            n_missing += 1
            continue
        actual = _sha256(p)
        if actual != exp_digest:
            failures.append(f"mask hash mismatch: {rel}")
            n_mismatch += 1
        n_checked += 1
        if n_checked % 500 == 0:
            print(f"  hashed {n_checked}/{len(expected)} masks...")
    print(f"  masks: {n_checked} checked, {n_missing} missing, "
          f"{n_mismatch} hash mismatches")
    return failures


def compare_snapshots() -> list[str]:
    failures: list[str] = []
    if not GOLDEN_SNAPSHOTS.exists():
        return []   # nothing to compare yet
    if not CURRENT_SNAPSHOTS.exists():
        return [f"missing current snapshots dir: {CURRENT_SNAPSHOTS}"]
    for golden_pkl in sorted(GOLDEN_SNAPSHOTS.glob("*.pkl")):
        cur = CURRENT_SNAPSHOTS / golden_pkl.name
        if not cur.exists():
            failures.append(f"snapshot missing: {golden_pkl.name}")
            continue
        try:
            g = pd.read_pickle(golden_pkl)
            c = pd.read_pickle(cur)
            if isinstance(g, pd.DataFrame):
                pd.testing.assert_frame_equal(c, g, check_dtype=False, atol=1e-8)
            elif isinstance(g, pd.Series):
                pd.testing.assert_series_equal(c, g, check_dtype=False, atol=1e-8)
            elif isinstance(g, np.ndarray):
                np.testing.assert_array_equal(c, g)
            else:
                if g != c:
                    failures.append(f"snapshot diff (scalar): {golden_pkl.name}")
        except AssertionError as e:
            failures.append(f"snapshot diff: {golden_pkl.name}\n    "
                            f"{str(e).splitlines()[0]}")
        except Exception as e:
            failures.append(f"snapshot read/compare error: {golden_pkl.name}: {e}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-masks", action="store_true",
                    help="skip mask hash comparison")
    ap.add_argument("--quick", action="store_true",
                    help="CSVs only (skip masks and snapshots)")
    args = ap.parse_args()

    all_failures: list[str] = []

    print("Comparing CSVs ...")
    all_failures += compare_csvs()

    if not args.quick:
        print("Comparing snapshots ...")
        all_failures += compare_snapshots()

    if not args.quick and not args.no_masks:
        print("Comparing mask hashes ...")
        all_failures += compare_mask_hashes()

    print()
    if all_failures:
        print(f"FAILED: {len(all_failures)} mismatches")
        for f in all_failures[:30]:
            print(f"  - {f}")
        if len(all_failures) > 30:
            print(f"  ... and {len(all_failures) - 30} more")
        return 1
    print("OK: every golden file matched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
