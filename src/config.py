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


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """Load config/default.yaml and normalize types.

    Conversions applied:
      min_photons_by_ncomp:  int / 'default' keys -> int / None
      tau_bounds[ft][param]: [lo, hi] list        -> (lo, hi) tuple
      tau_mean_bounds[ft]:   [lo, hi] list        -> (lo, hi) tuple (or None)
      taumean_cfg[ft]:       [[a, t], ...]        -> (('a', 'tau'), ...)
      ratio_cfg[ft]:         [num, den]           -> ('num', 'den')
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
