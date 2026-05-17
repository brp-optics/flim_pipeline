"""
make_master_metadata.py

Joins all per-file CSV outputs into results/master_metadata.csv.
One row per SDT file.  Run from the project root:

    uv run python make_master_metadata.py

Inputs read from results/:
    sdt_metadata_cal.csv      -- core metadata + phasor calibration values
    filepath_map.csv          -- absolute file paths
    position_annotation.csv   -- morphology annotation
    sdt_phasor_summary.csv    -- per-file G/S/tau phasor statistics
    fit_analysis_summary.csv  -- per-file tau_mean, amplitude ratio (best fit)
    fit_map.csv               -- fit-set / SDT file mapping (for best-fit selection)

Output:
    results/master_metadata.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "master_metadata.csv"

# Preferred number of fit components per fixation type -- mirrors Phase C/E config.
PREFERRED_N_COMP = {"glu": 3, "form": 2, "live": 2}


def _best_fit_key(filename: str, fixation_type, fit_map_df: pd.DataFrame):
    rows = fit_map_df[fit_map_df["sdt_filename"] == filename]
    if rows.empty:
        return None
    preferred = PREFERRED_N_COMP.get(str(fixation_type)) if fixation_type else None
    if preferred is not None and "n_components" in rows.columns:
        exact = rows[rows["n_components"] == preferred]
        if not exact.empty:
            return str(exact.iloc[0]["fit_set_key"])
    if "n_components" in rows.columns:
        return str(rows.sort_values("n_components", ascending=False).iloc[0]["fit_set_key"])
    return str(rows.iloc[0]["fit_set_key"])


def _load(name: str) -> pd.DataFrame | None:
    p = RESULTS_DIR / name
    if not p.exists():
        print(f"  [skip] {name} not found")
        return None
    df = pd.read_csv(p)
    print(f"  loaded {name}: {len(df)} rows, {len(df.columns)} cols")
    return df


def _merge_new_cols(base: pd.DataFrame, right: pd.DataFrame,
                    on: str, suffix: str = "_dup") -> pd.DataFrame:
    """Left-merge right into base, bringing only columns not already in base."""
    new_cols = [c for c in right.columns if c == on or c not in base.columns]
    return base.merge(right[new_cols], on=on, how="left")


def main():
    print("Building master_metadata.csv from results/ ...\n")

    # -- Base: one row per SDT file -----------------------------------------
    base = _load("sdt_metadata_cal.csv")
    if base is None:
        raise FileNotFoundError("sdt_metadata_cal.csv is required")

    # -- File paths ----------------------------------------------------------
    fp_map = _load("filepath_map.csv")
    if fp_map is not None:
        fp_map = fp_map.drop_duplicates(subset="filename", keep="first")
        base = _merge_new_cols(base, fp_map, on="filename")

    # Deduplicate base on filename (guard against Phase A fan-out)
    n_before = len(base)
    base = base.drop_duplicates(subset="filename", keep="first")
    if len(base) < n_before:
        print(f"  deduped base: dropped {n_before - len(base)} duplicate filename rows")

    # -- Position annotations -----------------------------------------------
    annot = _load("position_annotation.csv")
    if annot is not None:
        annot = annot.drop_duplicates(subset="filename", keep="first")
        base = _merge_new_cols(
            base,
            annot[["filename", "annotation", "notes"]].rename(
                columns={"annotation": "position_annotation",
                         "notes":      "position_notes"}
            ),
            on="filename",
        )

    # -- Phasor summary (per-file G/S/tau) -----------------------------------
    phasor = _load("sdt_phasor_summary.csv")
    if phasor is not None:
        phasor = phasor.drop_duplicates(subset="filename", keep="first")
        phasor_keep = [
            "filename",
            "n_px_total", "n_px_masked", "otsu_thresh",
            "G_cal_mean", "S_cal_mean", "G_cal_wmean", "S_cal_wmean",
            "G_cal_std",  "S_cal_std",  "M_wmean",
            "tau_phi_ns", "tau_mod_ns",
        ]
        phasor_keep = [c for c in phasor_keep if c in phasor.columns]
        base = _merge_new_cols(base, phasor[phasor_keep], on="filename")

    # -- Fit analysis summary (best fit per file) ----------------------------
    fit_analysis = _load("fit_analysis_summary.csv")
    fit_map_df   = _load("fit_map.csv")

    if fit_analysis is not None and fit_map_df is not None:
        # For each sample file, select the row from fit_analysis that corresponds
        # to the preferred fit model (PREFERRED_N_COMP).
        sample_rows = base[base["file_type"] == "sample"][
            ["filename", "fixation_type"]
        ].copy()

        best_keys = {
            row["filename"]: _best_fit_key(
                row["filename"], row.get("fixation_type"), fit_map_df
            )
            for _, row in sample_rows.iterrows()
        }

        # Build a filename -> best fit_analysis row mapping
        fit_analysis_indexed = fit_analysis.set_index("fit_set_key") \
            if "fit_set_key" in fit_analysis.columns else None

        if fit_analysis_indexed is not None:
            fit_rows = []
            for fn, key in best_keys.items():
                if key is None or key not in fit_analysis_indexed.index:
                    continue
                r = fit_analysis_indexed.loc[key].to_dict()
                r["filename"] = fn
                fit_rows.append(r)

            if fit_rows:
                fit_best = pd.DataFrame(fit_rows)
                fit_keep = [
                    "filename", "fit_set_key",
                    "n_px_total", "n_px_final", "pct_final",
                    "tau_mean_median_ps", "tau_mean_mean_ps", "tau_mean_std_ps",
                    "amp_ratio_median",   "amp_ratio_mean",   "amp_ratio_std",
                ]
                fit_keep = [c for c in fit_keep if c in fit_best.columns]
                # Rename columns that clash with phasor summary
                rename = {}
                for c in fit_keep:
                    if c != "filename" and c in base.columns:
                        rename[c] = f"fit_{c}"
                fit_best = fit_best[fit_keep].rename(columns=rename)
                base = _merge_new_cols(base, fit_best, on="filename")
                print(f"  fit summary: matched {len(fit_rows)} files")

    # -- Write ---------------------------------------------------------------
    base.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {len(base)} rows x {len(base.columns)} cols to {OUTPUT_FILE}")
    print(f"Columns:\n  {chr(10).join('  ' + c for c in base.columns)}")


if __name__ == "__main__":
    main()
