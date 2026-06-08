"""Load + normalize config/default.yaml into the dicts the library expects.

The YAML file uses lists where the library expects tuples and uses string
keys (like 'default') where the library expects None.  load_config() does
this normalization so notebook code can simply do:

    from src.config import load_config
    cfg = load_config()
    mask, brk = compute_fit_mask(
        fd, fixation_type, b_val,
        min_photons_by_ncomp=cfg['min_photons_by_ncomp'],
        chi2_lo=cfg['chi2_lo'], chi2_hi=cfg['chi2_hi'],
        tau_bounds=cfg['tau_bounds'],
        tau_mean_bounds=cfg['tau_mean_bounds'],
        taumean_cfg=cfg['taumean_cfg'],
    )
"""

from __future__ import annotations

from pathlib import Path

import yaml


_CONFIG_DIR          = Path(__file__).resolve().parent.parent / "config"
DEFAULT_CONFIG_PATH  = _CONFIG_DIR / "default.yaml"
DATA_DIRS_PATH       = _CONFIG_DIR / "data_dirs.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """Load config/default.yaml and normalize types to what the library expects.

    YAML lists become tuples and the string key 'default' becomes None so
    that downstream functions can use dict lookups without further conversion.

    Conversions applied:
      min_photons_by_ncomp: 'default' string key -> None; numeric string
                             keys ('1', '2', '3') -> int
      tau_bounds[ft][param]: [lo, hi] list -> (lo, hi) tuple
      tau_mean_bounds[ft]:   [lo, hi] list -> (lo, hi) tuple (or None)
      taumean_cfg[ft]:       [[a, t], ...] -> (('a', 'tau'), ...)
      ratio_cfg[ft]:         [num, den]    -> ('num', 'den')

    Parameters
    ----------
    path : Path or str, optional
        Path to a YAML config file.  Defaults to config/default.yaml in
        the project root (two directories above this file).

    Returns
    -------
    dict
        Normalized configuration dictionary.  Key entries:
          'min_photons_by_ncomp' -- {int|None: int}, threshold per n_components
          'chi2_lo', 'chi2_hi'   -- float, chi-squared acceptance window
          'tau_bounds'           -- {fixation_type: {param: (lo, hi)}}
          'tau_mean_bounds'      -- {fixation_type: (lo, hi) or None}
          'taumean_cfg'          -- {fixation_type: (('a1','tau1'), ...)}
          'ratio_cfg'            -- {fixation_type: ('numerator','denominator')}
          'data_dirs'            -- {'win': [...], 'lin': [...]}

    Side Effects
    ------------
    Reads the YAML file from disk.  Not cached; each call re-reads the file.

    Assumptions
    -----------
    The YAML file must exist and be parseable by yaml.safe_load.  Unknown
    top-level keys are passed through unmodified.

    Examples
    --------
    >>> cfg = load_config()
    >>> mask, brk = compute_fit_mask(
    ...     fd, fixation_type='form', b_val=2,
    ...     min_photons_by_ncomp=cfg['min_photons_by_ncomp'],
    ...     chi2_lo=cfg['chi2_lo'], chi2_hi=cfg['chi2_hi'],
    ...     tau_bounds=cfg['tau_bounds'],
    ...     tau_mean_bounds=cfg['tau_mean_bounds'],
    ... )

    Dependencies
    ------------
    yaml, pathlib
    """
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    out = dict(raw)   # shallow copy

    # min_photons_by_ncomp: 'default' -> None, '1'/'2'/'3' -> int
    mph = {}
    for k, v in raw.get("min_photons_by_ncomp", {}).items():
        if k == "default":
            mph[None] = v
        else:
            mph[int(k)] = v
    out["min_photons_by_ncomp"] = mph

    # tau_bounds: lists -> tuples
    tb = {}
    for ft, params in (raw.get("tau_bounds") or {}).items():
        tb[ft] = {p_: tuple(rng) for p_, rng in params.items()}
    out["tau_bounds"] = tb

    # tau_mean_bounds: list -> tuple, preserve None
    tmb = {}
    for ft, rng in (raw.get("tau_mean_bounds") or {}).items():
        tmb[ft] = tuple(rng) if rng is not None else None
    out["tau_mean_bounds"] = tmb

    # taumean_cfg: nested lists -> nested tuples
    tcfg = {}
    for ft, pairs in (raw.get("taumean_cfg") or {}).items():
        tcfg[ft] = tuple(tuple(pair) for pair in pairs)
    out["taumean_cfg"] = tcfg

    # ratio_cfg: list -> tuple
    rcfg = {}
    for ft, pair in (raw.get("ratio_cfg") or {}).items():
        rcfg[ft] = tuple(pair)
    out["ratio_cfg"] = rcfg

    return out


def save_data_dirs(
    win_dirs: list,
    lin_dirs: list,
    *,
    path: Path | str | None = None,
) -> Path:
    """Write data_dirs to config/data_dirs.yaml (called by Phase A).

    Phase A is the single authoritative source for which .sdt session
    directories belong to the pipeline.  Downstream phases read this file
    via get_data_dirs().  The file is completely overwritten on each call
    of this function.

    Parameters
    ----------
    win_dirs : list of str or Path
        Windows session-root paths, e.g.
        [r'E:\\18_RK_Circadian\\data\\raw\\20260429_KPC_fixed_dishes_on_SLIM'].
    lin_dirs : list of str or Path
        Linux session-root paths, e.g.
        ['/media/mint/BRPresbkup/18_RK_Circadian/data/raw/20260429_...'].
    path : Path or str, optional
        Destination YAML file.  Defaults to config/data_dirs.yaml.

    Returns
    -------
    Path
        Path to the file that was written.

    Side Effects
    ------------
    Creates or overwrites config/data_dirs.yaml.  Creates parent directories
    if they do not exist.

    Assumptions
    -----------
    Caller has write permission to the config/ directory.

    Examples
    --------
    >>> from src.config import save_data_dirs
    >>> win = [r'E:\\18_RK_Circadian\\data\\raw\\20260429_KPC_fixed_dishes_on_SLIM']
    >>> p = save_data_dirs(win, [])

    Dependencies
    ------------
    yaml, pathlib
    """
    out = Path(path) if path is not None else DATA_DIRS_PATH
    payload = {
        "data_dirs": {
            "win": [str(p) for p in win_dirs],
            "lin": [str(p) for p in lin_dirs],
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(
            "# Auto-generated by Phase A (notebooks/20260507_KPC_explore).\n"
            "# Do not hand-edit -- changes are overwritten on the next Phase A run.\n"
            "# Edit the WIN_DATA_DIRS / LIN_DATA_DIRS lists at the top of Phase A,\n"
            "# then re-run Phase A.  Downstream phases read this file via\n"
            "# src.config.get_data_dirs().\n\n"
        )
        yaml.safe_dump(payload, fh, sort_keys=False)
    return out


def get_data_dirs(
    config: dict | None = None,
    *,
    require_exist: bool = True,
) -> list[Path]:
    """Return raw .sdt session directories as Path objects.

    Reads from config/data_dirs.yaml (written by Phase A) if it exists,
    otherwise falls back to the data_dirs section of config/default.yaml.
    The win: and lin: lists are unioned, deduplicated (order-preserving),
    and optionally filtered to paths that exist on disk.

    Precedence:
      1. config/data_dirs.yaml  (written by Phase A; authoritative)
      2. config/default.yaml's data_dirs section (fallback for backwards
         compatibility before Phase A has been run on this checkout)

    Parameters
    ----------
    config : dict, optional
        Pre-loaded dict from load_config().  If None, both YAML files are
        read from disk.
    require_exist : bool, optional
        If True (default), raise FileNotFoundError when none of the listed
        paths exist on this machine.  Pass False to get the full list
        even when the data drive is not mounted.

    Returns
    -------
    list of Path
        Deduplicated, order-preserving list of session root directories.
        When require_exist=True, only paths that exist on disk are included.

    Raises
    ------
    FileNotFoundError
        When require_exist=True and no listed path exists on disk.

    Side Effects
    ------------
    Reads YAML files from disk.

    Assumptions
    -----------
    The win: and lin: lists may overlap (e.g. under WSL); duplicates are
    silently removed.

    Examples
    --------
    >>> data_dirs = get_data_dirs()
    >>> # On Linux with the drive mounted:
    >>> # [PosixPath('/media/mint/BRPresbkup/.../20260429_KPC_fixed_dishes_on_SLIM')]

    >>> # Allow offline use (drive not mounted):
    >>> all_dirs = get_data_dirs(require_exist=False)

    Dependencies
    ------------
    yaml, pathlib
    """
    # 1. Prefer data_dirs.yaml when present
    raw: dict = {}
    if DATA_DIRS_PATH.exists():
        with open(DATA_DIRS_PATH, encoding="utf-8") as fh:
            raw = (yaml.safe_load(fh) or {}).get("data_dirs", {}) or {}
    # 2. Fall back to default.yaml's data_dirs section
    if not raw:
        cfg = config if config is not None else load_config()
        raw = cfg.get("data_dirs", {}) or {}
    seen: set[str] = set()
    out:  list[Path] = []
    for os_key in ("win", "lin"):
        for d in raw.get(os_key, []) or []:
            p = Path(d)
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            if not require_exist or p.exists():
                out.append(p)
    if require_exist and not out:
        raise FileNotFoundError(
            f"None of the data_dirs in config exist on this machine.  "
            f"Listed: {list(seen)}"
        )
    return out
