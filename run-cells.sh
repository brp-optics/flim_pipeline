#!/bin/bash
set -euo pipefail

# Sample fitting routine for metabolic FLIM data from fixed cells.
# brp-optics with Claude Opus 4.6, 2026.03.09-11

# This script assumes:
#   - Data directories named *PCK_on_SLIM, each containing subdirectories
#     matching *exet-fitet-sz-b* (SPCImage biexponential fit exports).
#   - Each fit export directory contains per-position BH .asc files:
#     pos_NNNN_a1.asc, pos_NNNN_a2.asc, pos_NNNN_t1.asc, pos_NNNN_t2.asc,
#     pos_NNNN_chi.asc, pos_NNNN_photons.asc, pos_NNNN_color coded value.asc, etc.
#   - Python scripts (threshold-data.py, histogram-dir.py, histograms-to-stats.py,
#     asc-to-greyscale-tiff.py, crop-tif-dir.py) are on PATH or in current directory.
#   - fileutils.py is importable (same directory or on PYTHONPATH).
#
# Workflow overview:
#   1. Threshold & compute free-bound ratios (a1/a2) per fitting directory
#   2. Threshold & extract mean lifetime (color coded value) per fitting directory
#   3. Generate per-directory and combined histograms for QC
#   4. Convert thresholded data to greyscale TIFFs for visualization
#   5. Crop SPCImage .tif exports to 256x256
#
# Control slides (chroma, urea, pollen) are excluded from histograms since
# biexponential fits on single-fluorophore controls aren't meaningful.

# === Configuration ===
SCRIPTDIR="$(pwd)"
DATADIR="$1"
echo "DATADIR=$DATADIR"
PYTHON="uv run python"

# Suffix for threshold output files
TH_SUFFIX=".th.asc"

# Histogram bin settings (linear scale for tm; log scale for ar)
# These are passed to histogram-dir.py

# Greyscale TIFF display range -- set after inspecting histograms in step 3.
# Placeholder values for now.

AR_DISPLAY_MIN="${AR_DISPLAY_MIN:-0}"    # Free-bound ratio display min
AR_DISPLAY_MAX="${AR_DISPLAY_MAX:-5}"    # Free-bound ratio display max
TM_DISPLAY_MIN="${TM_DISPLAY_MIN:-0}"    # Mean lifetime display min (ps)
TM_DISPLAY_MAX="${TM_DISPLAY_MAX:-10000}"    # Mean lifetime display max (ps)

# Multiplication factor for greyscale TIFF (ImageJ doesn't handle fractional floats for ar.)
AR_MULTIPLY="${AR_MULTIPLY:-1000}"
TM_MULTIPLY="${TM_MULTIPLY:-1}"

# Exclusion patterns for control slides
EXCLUDE_PATTERNS="chroma|urea|pollen"

# === Helper functions ===
log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

# Find files matching suffix in a directory, excluding control slide patterns
# Usage: find_data_files <directory> <suffix>
find_data_files() {
    local dir="$1"
    local suffix="$2"
    find "$dir" -maxdepth 1 -name "*${suffix}" -type f \
	| grep -Ev "(${EXCLUDE_PATTERNS})" \
	|| true
    # Caveats: only works for one level of subdir.
}

# Extract the basename of a subdir relative to its parent, for naming outputs.
subdir_basename() {
    basename "$1"
}

dir_has_files() {
    [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]
}


echo "=== Run started at $(date) ===" | tee /dev/stderr
# =============
log "STEP 1 & 2: Threshold data and compute free-bound ratios + extract mean lifetime"

for dir in "$DATADIR"/*PCK_on_SLIM; do
    [ -d "$dir" ] || continue
    log "Processing experiment directory: $dir"
    
    for subdir in "$dir"/*exet-fitet-sz-b*; do
        [ -d "$subdir" ] || continue
        subdir_name="$(subdir_basename "$subdir")"
        log "  Fitting subdir: $subdir_name"

	BH_BIN="${subdir_name: -1}"
	# Flakey, breaks if BH_BIN > 9.
	
	# --- Step 1a: Threshold and compute free-bound ratios ---
        #  threshold-data.py loads all related BH exports for each position,
        #  applies quality thresholds (chi-sq, photons, lifetimes, amplitudes),
        #  computes a1/a2 ratio, and saves thresholded data.
        #
        # Output directory: $dir/t-ar-$subdir_name
        ar_outdir="$dir/t-ar-$subdir_name"
        mkdir -p "$ar_outdir"

        # Process each unique position (use a1 files as the anchor)
        ar_files=$(find_data_files "$subdir" "_a1.asc")
        if [ -z "$ar_files" ]; then
            log "    WARNING: No a1.asc files found in $subdir (skipping ar)"
        else
            n_files=$(echo "$ar_files" | wc -l)
            log "    Thresholding $n_files positions for free-bound ratio..."
            echo "$ar_files" | while IFS= read -r a1_file; do
                [ -f "$a1_file" ] || continue
                $PYTHON threshold-data.py "$a1_file" "$ar_outdir" \
                    --bh-bin "$BH_BIN" --suffix "$TH_SUFFIX"
            done
            log "    Free-bound ratio results saved in $ar_outdir"
        fi

	echo "=== Step 1b started at $(date) ===" | tee /dev/stderr
        # -- Step 1b: Threshold and extract mean lifetime --
        # Same thresholding, but we specifically want the "color coded value"
        # files which contain t_m (mean lifetime).
        #
        # Output directory: $dir/t-tm-$subdir_name
        tm_outdir="$dir/t-tm-$subdir_name"
        mkdir -p "$tm_outdir"

        tm_files=$(find_data_files "$subdir" "_color coded value.asc")
        if [ -z "$tm_files" ]; then
            log "    WARNING: No 'color coded value' files found in $subdir (skipping tm)"
        else
            n_files=$(echo "$tm_files" | wc -l)
            log "    Thresholding $n_files positions for mean lifetime..."
            echo "$tm_files" | while IFS= read -r ccv_file; do
                [ -f "$ccv_file" ] || continue
                $PYTHON threshold-data.py "$ccv_file" "$tm_outdir" \
                    --bh-bin "$BH_BIN" --suffix "$TH_SUFFIX"
            done
	    log "    Mean lifetime results in: $tm_outdir"
        fi

	echo "=== Step 2a started at $(date) ===" | tee /dev/stderr
        # -- Step 2a: Per-directory histograms for free-bound ratio --
        log "    Generating per-directory ar histograms..."

        # NADH channel (457 nm excitation)
        $PYTHON histogram-dir.py "$ar_outdir" \
            --suffix "_ar${TH_SUFFIX}" \
            --include "457" \
            --log \
            --zero_cutoff 0.01 \
            --saveplot "$ar_outdir/hist-457-ar.png" \
            --title "Free-bound ratio (NADH, 457) - $subdir_name" \
            --text-output "$ar_outdir/hist-457-ar-stats.txt" \
            || log "      (no NADH ar files or histogram failed)"

        # FAD channel (535 nm excitation)
        $PYTHON histogram-dir.py "$ar_outdir" \
            --suffix "_ar${TH_SUFFIX}" \
            --include "535" \
            --log \
            --zero_cutoff 0.01 \
            --saveplot "$ar_outdir/hist-535-ar.png" \
            --title "Free-bound ratio (FAD, 535) - $subdir_name" \
            --text-output "$ar_outdir/hist-535-ar-stats.txt" \
            || log "      (no FAD ar files or histogram failed)"

	echo "=== Step 2b at $(date) ===" | tee /dev/stderr
        # -- Step 2b: Per-directory histograms for mean lifetime --
        log "    Generating per-directory tm histograms..."

        # NADH channel
        $PYTHON histogram-dir.py "$tm_outdir" \
            --suffix "_color coded value${TH_SUFFIX}" \
            --include "457" \
            --saveplot "$tm_outdir/hist-457-tm.png" \
            --title "Mean lifetime (NADH, 457) - $subdir_name" \
            --text-output "$tm_outdir/hist-457-tm-stats.txt" \
            || log "      (no NADH tm files or histogram failed)"

        # FAD channel
        $PYTHON histogram-dir.py "$tm_outdir" \
            --suffix "_color coded value${TH_SUFFIX}" \
            --include "535" \
            --saveplot "$tm_outdir/hist-535-tm.png" \
            --title "Mean lifetime (FAD, 535) - $subdir_name" \
            --text-output "$tm_outdir/hist-535-tm-stats.txt" \
            || log "      (no FAD tm files or histogram failed)"

    done
done

# ==========
# == STEP 3: Combined histograms across all experiment directories ==
# ==========

echo "=== Step 3 started at $(date) ===" | tee /dev/stderr
log "=== STEP 3: Combined histograms across all experiments ==="

mkdir -p combined_histograms

# == Combined free-bound ratio histograms ==================================

# NADH ar (457)
log "  Combined NADH free-bound ratio histogram..."
$PYTHON histogram-dir.py "$DATADIR" \
    --suffix "_ar${TH_SUFFIX}" \
    --include "457" \
    --recursive \
    --log \
    --zero_cutoff 0.01 \
    --saveplot "combined_histograms/combined-457-ar.png" \
    --title "Combined Free-bound ratio (NADH, 457)" \
    --text-output "combined_histograms/combined-457-ar-stats.txt" \
    || log "  (combined NADH ar histogram failed or no files)"

# FAD ar (535)
log "  Combined FAD free-bound ratio histogram..."
$PYTHON histogram-dir.py "$DATADIR" \
    --suffix "_ar${TH_SUFFIX}" \
    --include "535" \
    --recursive \
    --log \
    --zero_cutoff 0.01 \
    --saveplot "combined_histograms/combined-535-ar.png" \
    --title "Combined Free-bound ratio (FAD, 535)" \
    --text-output "combined_histograms/combined-535-ar-stats.txt" \
    || log "  (combined FAD ar histogram failed or no files)"

echo "=== Combined mean lifetime histograms started at $(date) ===" | tee /dev/stderr
# == Combined mean lifetime histograms =====================================

# NADH tm (457)
log "  Combined NADH mean lifetime histogram..."
$PYTHON histogram-dir.py "$DATADIR" \
    --suffix "_color coded value${TH_SUFFIX}" \
    --include "457" \
    --recursive \
    --saveplot "combined_histograms/combined-457-tm.png" \
    --title "Combined Mean lifetime (NADH, 457)" \
    --text-output "combined_histograms/combined-457-tm-stats.txt" \
    || log "  (combined NADH tm histogram failed or no files)"

# FAD tm (535)
log "  Combined FAD mean lifetime histogram..."
$PYTHON histogram-dir.py "$DATADIR" \
    --suffix "_color coded value${TH_SUFFIX}" \
    --include "535" \
    --recursive \
    --saveplot "combined_histograms/combined-535-tm.png" \
    --title "Combined Mean lifetime (FAD, 535)" \
    --text-output "combined_histograms/combined-535-tm-stats.txt" \
    || log "  (combined FAD tm histogram failed or no files)"

log ""
log "====================================================================="
log "=  CHECKPOINT: Review histograms in combined_histograms/ and each    ="
log "=  t-ar-*/t-tm-*/ directory. Update AR_DISPLAY_MIN/MAX and           ="
log "=  TM_DISPLAY_MIN/MAX below, then re-run from step 4 onward.        ="
log "=                                                                     ="
log "=  Current display range settings:                                    ="
log "=    AR: [$AR_DISPLAY_MIN, $AR_DISPLAY_MAX]  (${AR_MULTIPLY} for TIFF)            ="
log "=    TM: [$TM_DISPLAY_MIN, $TM_DISPLAY_MAX] ps (${TM_MULTIPLY} for TIFF)         ="
log "======================================================================"
log ""
log "To skip to step 4, set SKIP_TO_TIFF=1 and re-run."

if [ "${SKIP_TO_TIFF:-0}" != "1" ]; then
    read -rp "Press Enter to continue to greyscale TIFF generation (or Ctrl-C to stop and adjust thresholds)..."
fi

echo "=== Step 4 started at $(date) ===" | tee /dev/stderr
# ==============================================================================
# STEP 4: Generate greyscale TIFFs from thresholded data
# ==============================================================================

log "=== STEP 4: Generating greyscale TIFFs ==="

# == Free-bound ratio TIFFs ================================================
for subdir in "$DATADIR"/*PCK_on_SLIM/t-ar-*; do
    [ -d "$subdir" ] || continue
    dir="$(dirname "$subdir")"
    subdir_name="$(subdir_basename "$subdir")"
    gs_outdir="$dir/gs-$subdir_name"
    mkdir -p "$gs_outdir"

    log "  Generating ar greyscale TIFFs: $subdir -> $gs_outdir"
    $PYTHON asc-to-greyscale-tiff.py "$subdir" \
        --insuffix "_ar${TH_SUFFIX}" \
        --outdir "$gs_outdir" \
        --outsuffix "_ar_f32.tif" \
        --multiplyby "$AR_MULTIPLY"

    # Manual inspection
    log "  Opening $gs_outdir for inspection..."
    open "$gs_outdir" 2>/dev/null || true
done

# == Mean lifetime TIFFs ===================================================
for subdir in "$DATADIR"/*PCK_on_SLIM/t-tm-*; do
    [ -d "$subdir" ] || continue
    dir="$(dirname "$subdir")"
    subdir_name="$(subdir_basename "$subdir")"
    gs_outdir="$dir/gs-$subdir_name"
    mkdir -p "$gs_outdir"

    log "  Generating tm greyscale TIFFs: $subdir -> $gs_outdir"
    # For tm, the thresholded "color coded value" files have the tm values.
    $PYTHON asc-to-greyscale-tiff.py "$subdir" \
        --insuffix "_color coded value${TH_SUFFIX}" \
        --outdir "$gs_outdir" \
        --outsuffix "_tm_f32.tif" \
        --multiplyby "$TM_MULTIPLY"

    # Manual inspection
    log "  Opening $gs_outdir for inspection..."
    open "$gs_outdir" 2>/dev/null || true
done

# ==============================================================================
# STEP 5: Crop SPCImage .tif exports to 256x256
# ==============================================================================
echo "=== Step 5 started at $(date) ===" | tee /dev/stderr
log "=== STEP 5: Cropping SPCImage .tif exports to 256x256 ==="

for dir in "$DATADIR"/*PCK_on_SLIM; do
    [ -d "$dir" ] || continue

    for subdir in "$dir"/*exet600-1400-0-500-fitet-sz-b*; do
        [ -d "$subdir" ] || continue
        subdir_name="$(subdir_basename "$subdir")"
        crop_outdir="$dir/cropped-$subdir_name"
        mkdir -p "$crop_outdir"

        log "  Cropping: $subdir -> $crop_outdir"
        $PYTHON crop-tif-dir.py "$subdir" \
		"$crop_outdir" \
		--shape 256x256

        # Manual inspection: compare original, cropped, and thresholded tm
        # Find matching t-tm directory for side-by-side comparison
        # (strip the "exet600-1400-0-500-" prefix to match the base fitting dir name)
        log "  Opening for comparison..."
        open "$subdir" 2>/dev/null || true
        open "$crop_outdir" 2>/dev/null || true
        # Try to find matching tm directory
        for tm_dir in "$dir"/t-tm-*; do
            [ -d "$tm_dir" ] && open "$tm_dir" 2>/dev/null || true
            break  # just open the first match for reference
        done
    done
done

log ""
log "=== Pipeline complete ==="
log "Summary of outputs:"
log "  t-ar-*/         : Thresholded free-bound ratio .asc files"
log "  t-tm-*/         : Thresholded mean lifetime .asc files"
log "  gs-t-ar-*/      : Greyscale TIFF free-bound ratio images"
log "  gs-t-tm-*/      : Greyscale TIFF mean lifetime images"
log "  cropped-*/      : Cropped 256x256 SPCImage exports"
log "  combined_histograms/ : Combined histograms across all experiments"


    # for subdir in "$dir/*exet-fitet-sz-b*"
    # do

	# Generate histogram of the entire t-ar- directory
	# Do not include files with "chroma", "urea", or "pollen" in the name (control slide biexponentials aren't meaningful.)
	# Histogram files containing 535 in the name and 457 in the name separately (NADH vs FAD)
	# Display histograms and proposed thresholds for display


	# Threshold "-color-coded-value.asc" files (these contain mean lifetime t_m), in the same way that the Free-bound ratios were thresholded.
	# Results are saved in $dir/t-tm-$subdir

	# Generate histogram of the entire t-tm- directory
	# Do not include files with "chroma", "urea", or "pollen" in the name (control slide biexponentials aren't meaningful.)
	# Histogram files containing 535 in the name and 457 in the name separately (NADH vs FAD)
	# Display histograms and proposed thresholds for display
	
#    done

