# FLIM analysis pipeline

## Phase A — Inventory and metadata

1. **Discover all files.**
   Recursively scan the data directory for `.sdt`, `.spc`, and ascii fit-export files.
   → *raw file list*

2. **Build metadata DataFrame.**
   Parse sdt headers to extract `acquisition_time`, scan dimensions, bin width, rep rate. Tag each file with `file_type` (IRF-before, IRF-after, chroma-before, chroma-after, sample), `cell_appearance`, and `colony_size`. Match each sdt with its corresponding ascii fit export.
   → *df with one row per acquisition, sorted by acquisition\_time*

3. **Load ascii fit exports.**
   Parse a1, a2, a3, tau1, tau2, tau3, photon count, χ² per pixel from ascii files. Store as numpy arrays keyed by filepath stem.
   → *dict of fit arrays per file*

## Phase B — Instrument calibration

4. **Compare bracketing IRFs.**
   Load before and after IRF sdt files. Overlay decay curves. Quantify peak channel position, FWHM, and integrated counts for each. Compute drift (Δpeak, ΔFWHM) between the two.
   → *IRF drift metrics; decision on interpolation vs averaging*

   > ⚠ If peak shift > 1 time bin: flag session, consider per-file IRF interpolation by acquisition\_time.

5. **Compute phasor calibration from chroma slides.**
   Load before/after chroma sdt files. Compute raw phasor (G, S) for each. Compare against the theoretical position for the known fluorophore lifetime at ω. Derive phase rotation and modulation correction factors. Check whether calibration drifted between before/after.
   → *phasor\_cal\_phase, phasor\_cal\_mod (per calibration timepoint)*

6. **Assign calibration to each sample.**
   For each sample file, use its `acquisition_time` to determine which IRF and phasor calibration to apply (nearest, averaged, or interpolated). Record `irf_source` and `phasor_cal_source` columns in df.
   → *df updated with calibration provenance columns*

## Phase C — Data integrity checks

7. **Verify orientation match between sdt and fit exports.**
   For a few representative files, generate an intensity image from the sdt (sum over time bins) and compare against the photon count map from the ascii export. Check for flips (LR, UD) and transposition. Determine the transform needed, if any.
   → *orientation\_transform (None, "flipud", "fliplr", "transpose", etc.)*

8. **Spot-check exported fits against raw data.**
   Select 3–5 bright pixels from a representative file. Fit mono-exponential from the raw sdt decay using your own fitter. Compare recovered τ and amplitude against the corresponding ascii export values. Verify agreement within tolerance.
   → *confidence that ascii exports are trustworthy (or a list of discrepancies)*

## Phase D — Phasor analysis

9. **Compute intensity mask.**
   For each sdt file, sum photon counts over time bins per pixel. Threshold to create a boolean mask excluding background pixels. Record `fraction_above_threshold` in df.
   → *mask array per file*

10. **Compute calibrated phasor per pixel.**
    For each sdt file, compute raw (G, S) at the laser rep frequency using phasorpy. Apply the calibration correction from step 5. Store calibrated (G, S) arrays. Apply intensity mask before any summary statistics.
    → *G, S arrays per file; G\_mean, S\_mean columns in df*

11. **Phasor sanity check.**
    Overlay phasor cloud on the universal semicircle. Verify the chroma reference point lands at its expected position after calibration. Check whether sample points fall on, inside, or outside the semicircle (indicating single-exp, multi-exp, or calibration error respectively).
    → *phasor semicircle plot; flag files with points outside the semicircle*

12. **Plot phasors grouped by file type.**
    Generate one phasor plot per group (`file_type` / `cell_appearance` / `colony_size`). Color-code by group for quick visual feedback to collaborator.
    → *phasor summary figures for sharing*

## Phase E — Exported fit analysis

13. **Apply orientation correction to fit arrays.**
    Using the transform determined in step 7, flip/transpose the ascii fit arrays so they align pixel-for-pixel with the sdt data and phasor arrays.
    → *aligned fit arrays*

14. **Threshold on fit quality.**
    Create a fit-quality mask from reduced χ² (e.g. 0.8 < χ² < 1.5). Optionally intersect with the intensity mask from step 9. Inspect the χ² map spatially for systematic patterns.
    → *fit\_quality\_mask per file; chi2\_mean column in df*

15. **Compute derived lifetime metrics.**
    From the masked fit parameters, compute amplitude-weighted mean lifetime per pixel: τ\_mean = Σ(a\_i · τ\_i) / Σ(a\_i). Compute fractional contributions f\_i = a\_i · τ\_i / Σ(a\_j · τ\_j). Add per-image summary statistics to df.
    → *tau\_mean array per file; tau\_mean\_mean, f1\_mean, f2\_mean columns in df*

## Phase F — Group comparisons

16. **Check for photobleaching.**
    If sdt files contain frame data, compare early vs late frame intensity. Flag files with significant bleaching (>10% decline) in df.
    → *bleaching\_flag, bleaching\_pct columns in df*

17. **Intra-group distributions.**
    For each group, plot per-pixel lifetime distributions (histograms or KDEs) from each image overlaid. Assess within-group consistency: do images from the same condition look similar?
    → *intra-group overlay plots*

18. **Inter-group comparisons.**
    Compare groups using per-image summary statistics (one value per image, not per pixel). Violin or box plots of τ\_mean, fractional contributions, phasor position. Statistical tests with the image (or colony) as the unit of analysis, not the pixel.
    → *inter-group comparison figures; statistical test results*

19. **Cross-validate phasor vs fit results.**
    Compare phasor-derived lifetimes (τ\_phi, τ\_mod from step 10) against fit-derived mean lifetimes (step 15) on the same pixels. Agreement strengthens confidence; disagreement highlights model mismatch or calibration issues.
    → *phasor-vs-fit correlation plot; Bland-Altman plot*

20. **Export results.**
    Save the final df as CSV. Export lifetime maps and phasor arrays as TIFF or HDF5. Package summary figures for collaborator.
    → *results/ directory with CSV, TIFFs, figures*
