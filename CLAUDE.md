# flim-pipeline

FLIM (Fluorescence Lifetime Imaging Microscopy) analysis pipeline for KPC pancreatic cancer
cell data acquired on the SLIM instrument (Becker & Hickl TCSPC). Collaborator: Bjorn Paulson.

## Project layout

```
flim_pipeline/
├── notebooks/
│   ├── 20260507_KPC_explore.py   # jupytext source (edit this)
│   └── 20260507_KPC_explore.ipynb # auto-generated — do not edit directly
├── results/                       # created at runtime
│   ├── sdt_metadata.csv
│   ├── sdt_metadata.xlsx          # browsable metadata for manual verification
│   └── filepath_map.csv
├── Makefile                       # `make notebooks` syncs .py → .ipynb
├── pyproject.toml
└── requirements-frozen.txt
```

## Notebook workflow

Notebooks are jupytext percent-format `.py` files. Edit `.py` in Emacs or Claude Code,
then regenerate `.ipynb`:

```bash
# In git-bash from the project root:
make notebooks
# Equivalent: uv run jupytext --sync notebooks/*.py
```

The jupytext format pairing is configured in `pyproject.toml`:
```toml
[tool.jupytext]
formats = "notebooks//py:percent,notebooks//ipynb"
```

Never edit the `.ipynb` directly — changes will be overwritten by the next sync.

## Environment

```bash
uv sync --dev       # install all deps including jupyterlab, jupytext, openpyxl
uv run jupyter lab  # launch JupyterLab
```

Dependencies are in `pyproject.toml`. Key packages: `sdtfile`, `pandas`, `numpy`,
`matplotlib`, `scipy`. `pyqt5` is Linux-only (Windows bundles Qt in the wheel;
`pyqt5-qt5` does not exist on PyPI for Windows).

## Data

**Instrument:** Becker & Hickl SPCM, `.sdt` files (TCSPC image stacks).
**Fit exports:** SPCImage exports per-pixel fit parameters as `.asc` files (space-delimited 2D grids, one parameter per file).

**Raw data directories:**

| OS | Path |
|----|------|
| Linux | `/media/mint/BRPresbkup/18_RK_Circadian/data/raw/` |
| Windows | `E:\18_RK_Circadian\data\raw\` |

**Sessions:**
- `20260429_KPC_fixed_dishes_on_SLIM`
- `20260501_KPC_fixed_dishes_on_SLIM`

OS is selected at the top of the notebook via `current_os = "Win"` / `"Linux"`.

## Filename convention

```
{index}_{sample}_{wavelength}nm_{power}mW_u{usb}_{pockels}_{objective}_{em}s{bw}_g{gain}_z{z}_{pixels}pix_{mode}_{frames}f_{frame_idx:04d}.sdt
```

Example fields parsed by `parse_filename_metadata()`:
- `wavelength_nm` — excitation (e.g. 740)
- `power_mW` — laser power
- `usb_atten` — USB attenuator setting
- `pockels` — Pockels cell value (`poc0p2` → 0.2)
- `objective` — e.g. `20x0.75NA`
- `em_filter_nm` / `em_bandwidth_nm` — emission filter (e.g. `457s50` → 457 nm ± 25)
- `gain` — detector gain
- `cell_type` — `KPCWT`, `BKO`, `DKO`, or `KPC`
- `fixation_type` — `glu`, `form`, `live`, `chromablue`, `urea`, `pollen`

## Pipeline phases

### Phase A — Inventory and metadata (COMPLETE in `20260507_KPC_explore.py`)

1. Discover `.sdt` and `.asc` files → `files_df`
2. Parse filename metadata → `sdt_df` with instrument and sample columns
3. Classify files: `chroma`, `urea_reference`, `pollen_reference`, `sample`, `other`
4. Load SDT file headers (timestamp, scan dims, photon stats, time axis) → merged into `sdt_df`
5. Load and group `.asc` fit exports → `fit_data` dict (`{base_stem: {param: 2D array}}`)
6. Match fit exports to SDT files by stem similarity
7. Sanity check: load one file end-to-end, verify shapes
8. Save `sdt_metadata.csv`, `sdt_metadata.xlsx`, `filepath_map.csv`
9. **Verification cells:** print raw BH header snippets vs parsed fields for 5 sampled files

Key outputs: `sdt_df` (one row per SDT file), `fit_data` (dict of fit arrays),
`load_decay(filepath)`, `get_fit_arrays(row)`, `get_row(df, filename)`.

### Phases B–F — PLANNED (not yet implemented)

- **B** — Instrument calibration: IRF comparison, phasor calibration from chroma slides
- **C** — Data integrity: orientation match between SDT and fit exports, spot-check fits
- **D** — Phasor analysis: intensity masks, calibrated (G, S) per pixel, semicircle plots
- **E** — Fit analysis: orientation correction, chi² masking, amplitude-weighted τ_mean
- **F** — Group comparisons: photobleaching check, intra/inter-group distributions, phasor vs fit cross-validation

## Key design notes

- **No IRF files found** in the two sessions so far. IRF acquisition may be in a separate
  directory or use a different naming scheme. A manual override is needed:
  `sdt_df.loc[sdt_df["filename"].str.contains("pattern"), "file_type"] = "irf_before"`

- **BH timestamp parsing** (`_parse_bh_timestamp`): reads `*DATE` / `*TIME` lines from the
  info text block in the SDT header. Falls back to filesystem mtime. Check
  `time_delta_s` column to verify agreement.

- **Fit export path keys** use `_path_{param}` prefix (e.g. `_path_a1`) so they are excluded
  from analysis by checking `not k.startswith("_")`.

- **`save_cols`** excludes `filepath` (not portable across machines) and `info_text_snippet`.
  The `filepath_map.csv` keeps the path separately for re-joining on the same machine.
