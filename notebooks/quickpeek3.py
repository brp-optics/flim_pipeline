# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # quickpeek3
# Like-to-like comparison of form and live files at matched imaging parameters.

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path

# %%
sdt = pd.read_csv(r'..\results\sdt_metadata_cal.csv')
df  = pd.read_csv(r'..\results\fit_analysis_summary.csv')

# Bring in pockels from sdt_df only if Phase E didn't already write it
if 'pockels' not in df.columns:
    df = df.merge(sdt[['filename', 'pockels']], on='filename', how='left')

# Same for fit_mask_path
if 'fit_mask_path' not in df.columns:
    DATA_ROOT = Path(r'E:\18_RK_Circadian\data\raw')
    def _mask_path(key):
        s, r, b = key.split('::', 2)
        return str(DATA_ROOT / s / r / f'{b}_fit_mask.npy')
    df['fit_mask_path'] = df.fit_set_key.apply(_mask_path)

# treatment_duration also needed for the form 10min filter
if 'treatment_duration' not in df.columns:
    df = df.merge(sdt[['filename', 'treatment_duration']], on='filename', how='left')

df['date'] = pd.to_datetime(df.session_root.str.split('_').str[0], format='%Y%m%d')
df['PC']   = ['High' if p > 0.3 else 'Low' for p in df.pockels]

# %%
annot = pd.read_csv(r'..\results\position_annotation.csv')
df = df.merge(annot[['filename', 'annotation']], on='filename', how='left')
df['annotation'] = df['annotation'].fillna('(unannotated)')

# %% [markdown]
# ## Step 1: file counts by parameter combination

# %%
df.groupby(['fixation_type', 'cell_type', 'em_filter_nm']).size()

# %%
df.groupby(['fixation_type', 'cell_type', 'em_filter_nm', 'PC']).size()

# %%
df.groupby(['fixation_type', 'date', 'cell_type', 'em_filter_nm', 'PC', 'annotation']).size()

# %% [markdown]
# ## Step 2: pick matched subsets and display photons grids

# %%
FORM = df.loc[(df.fixation_type=='form') & (df.em_filter_nm==457)
              & (df.PC=='Low') & (df.annotation=='colony_deep')] \
         .sort_values(['date', 'cell_type', 'filename']).reset_index(drop=True)
LIVE = df.loc[(df.fixation_type=='live') & (df.em_filter_nm==457) & (df.PC=='Low')] \
         .sort_values(['date', 'cell_type', 'filename']).reset_index(drop=True)
print('FORM:'); print(FORM.groupby(['date', 'cell_type']).size())
print('\nLIVE:'); print(LIVE.groupby(['date', 'cell_type']).size())


# %%
def show_photons(subset, ncols=4, title=''):
    n = len(subset)
    if n == 0:
        print(f'(empty: {title})'); return
    nrows = -(-n // ncols)
    plt.figure(figsize=(5.5*ncols, 4.5*nrows))
    for i, (_, row) in enumerate(subset.iterrows()):
        print(row.fit_mask_path)
        ph_path = row.fit_mask_path.replace('_fit_mask.npy', '_photons.asc')
        ax = plt.subplot(nrows, ncols, i+1)
        try:
            ph = np.loadtxt(ph_path)
            finite = ph[ph > 0]
            vlo = float(np.percentile(finite, 1))  if finite.size else 0.0
            vhi = float(np.percentile(finite, 99)) if finite.size else 1.0
            plt.imshow(ph, cmap='inferno', vmin=vlo, vmax=vhi)
            plt.colorbar(fraction=0.046, pad=0.04)
        except Exception:
            plt.text(0.5, 0.5, 'no .asc', ha='center', transform=ax.transAxes)
        wrapped = '\n'.join(textwrap.wrap(row.filename, width=55))
        plt.title(f"{row.cell_type} {row.date.strftime('%m-%d')}\n{wrapped}", fontsize=7)
        plt.axis('off')
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

show_photons(FORM, title='FORM colony_deep, 457nm, Low PC')
show_photons(LIVE, title='LIVE 457nm, Low PC')

# %% [markdown]
# ## Step 3: pick matched pairs (manual)
# ## Step 4: check consistency (later)

# %%
PAIRS = [
    ('form 20min  5/9',
     r'E:\18_RK_Circadian\data\raw\20260509_KPC_fixed_dishes_on_SLIM\fitet-sz-c2-b10\3_KPCWTform20min_740nm_3006mW_u1_poc0p25_20x0p75NA_475s50_g70_z1_256pix_basicFLIM_120f_0001_fit_mask.npy',
     r'E:\18_RK_Circadian\data\raw\20260509_KPC_fixed_dishes_on_SLIM\fitet-sz-c2-b10\4_BKOCO2form20min_740nm_2995mW_u1_poc0p25_20x0p75NA_457s50_g70_z1_256pix_basicFLIM_120f_0004_fit_mask.npy'),
    ('form 10min  5/20',
     r'E:\18_RK_Circadian\data\raw\20260520_KPC_fixed_dishes_SLIM\fitet-sz-c2-b9\3_KPCWTform10min_740nm_3082mW_u1_poc0p25_20x0p75NA_457s50_g70_z1_256pix_basicFLIM_120f_0010_fit_mask.npy',
     r'E:\18_RK_Circadian\data\raw\20260520_KPC_fixed_dishes_SLIM\fitet-sz-c2-b9\4_BKOCO2form10min_740nm_3099mW_u1_poc0p25_20x0p75NA_457s50_g70_z1_256pix_basicFLIM_120f_0006_fit_mask.npy'),
    ('live  5/8 vs 5/17',
     r'E:\18_RK_Circadian\data\raw\20260508_KPC_live_on_SLIM\fitet-sz-c2-b5\3_KPCWTlive_740nm_3033mW_u1_poc0p2_20x0p75NA_475s50_g70_z1_256pix_basicFLIM_120f_0011_fit_mask.npy',
     r'E:\18_RK_Circadian\data\raw\20260517_KPC_live_on_SLIM\fitet-sz-c2-b5\3_BKOCO2live_740nm_3097mW_u1_poc0p2_20x0p75NA_457s50_g70_z1_256pix_basicFLIM_120f_0015_fit_mask.npy'),
    ('live  5/1 vs 5/17',
     r'E:\18_RK_Circadian\data\raw\20260501_KPC_fixed_dishes_on_SLIM\batch2\fitet-sz-c2-b10\15_KPC_live_posn3_740nm_3040mW_u1_poc0p2_20x0p75NA_457s50_g70_z1_256pix_basicFLIM_120f_0002_fit_mask.npy',
     r'E:\18_RK_Circadian\data\raw\20260517_KPC_live_on_SLIM\fitet-sz-c2-b5\3_BKOCO2live_740nm_3097mW_u1_poc0p2_20x0p75NA_457s50_g70_z1_256pix_basicFLIM_120f_0008_fit_mask.npy'),
]

def label_of(stem):
    s = stem.upper()
    cell = 'BKO' if 'BKO' in s else 'KPCWT'
    if   'LIVE' in s:       cond = 'live'
    elif 'FORM20MIN' in s:  cond = 'form 20min'
    elif 'FORM10MIN' in s:  cond = 'form 10min'
    elif 'FORM' in s:       cond = 'form'
    elif 'GLU' in s:        cond = 'glu'
    else:                   cond = '?'
    return f'{cell} {cond}'

def load_all(mask_path):
    p = Path(mask_path)
    folder = p.parent
    stem = p.name.replace('_fit_mask.npy', '')
    arrs = {
        'photons':  np.loadtxt(folder / f'{stem}_photons.asc'),
        'tau_mean': np.loadtxt(folder / f'{stem}_color coded value.asc'),
        'a1':       np.loadtxt(folder / f'{stem}_a1.asc'),
        'a2':       np.loadtxt(folder / f'{stem}_a2.asc'),
        'chi2':     np.loadtxt(folder / f'{stem}_chi.asc'),
        'mask':     np.load(p),
    }
    arrs['a1/a2'] = np.where(arrs['a2'] > 0, arrs['a1']/arrs['a2'], np.nan)
    return arrs, stem

def show_row(ax_row, arrs):
    mask = arrs['mask']
    panels = [('photons','inferno'), ('tau_mean','RdBu_r'),
              ('a1','viridis'), ('a2','viridis'),
              ('a1/a2','RdBu_r'), ('chi2','viridis')]
    for ax, (key, cmap_name) in zip(ax_row[:6], panels):
        arr = arrs[key]
        # photons shown raw; everything else masked so rejected px are hidden
        display = arr if key == 'photons' else np.where(mask, arr, np.nan)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad('lightgray')
        finite = display[np.isfinite(display)]
        vlo = float(np.percentile(finite, 1))  if finite.size else 0.0
        vhi = float(np.percentile(finite, 99)) if finite.size else 1.0
        im = ax.imshow(display, cmap=cmap, vmin=vlo, vmax=vhi)
        inside = arr[mask & np.isfinite(arr)]
        i_m = float(inside.mean()) if inside.size else float('nan')
        ax.set_title(f'{key}\n{i_m:.2f}', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
    ax = ax_row[6]
    im = ax.imshow(mask.astype(float), cmap='gray_r', vmin=0, vmax=1)
    ax.set_title(f'mask\n{int(mask.sum())} px ({100*mask.mean():.0f}%)', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')

fig, axes = plt.subplots(8, 7, figsize=(28, 28))
fig.subplots_adjust(hspace=0.6, wspace=0.35, left=0.08)
for i, (title, p1, p2) in enumerate(PAIRS):
    for j, p in enumerate([p1, p2]):
        ri = i*2 + j
        arrs, stem = load_all(p)
        lab = label_of(stem)
        show_row(axes[ri], arrs)
        axes[ri][0].set_ylabel(f'[{title}]\n{lab}', fontsize=11, fontweight='bold',
                                rotation=0, labelpad=90, va='center', ha='right')
plt.suptitle('4 pairs: photons | tau_mean | a1 | a2 | a1/a2 | chi2 | mask  '
             '(rejected px = gray; titles = mean inside mask)', fontsize=11)
plt.show()

# %%
# Recompute the .npy quality mask for just the 8 PAIRS files using the
# new (lower) photon threshold.  Overwrites the existing _fit_mask.npy.
# Same criteria as Phase E:
#   1. uniform_filter(photons, 2b+1) * (2b+1)^2  >=  MIN_PH
#   2. CHI2_LO  <= chi^2 <= CHI2_HI
#   3. a1, a2 >= 0 and a1+a2 > 0
#   4. tau1 in [50, 1500] ps, tau2 in [800, 6000] ps
#   5. live only: tau_mean >= 250 ps
import re
from scipy.ndimage import uniform_filter

MIN_PH                  = 1000
CHI2_LO, CHI2_HI        = 0.8, 2.0
TAU1_LO, TAU1_HI        = 50.0, 1500.0
TAU2_LO, TAU2_HI        = 800.0, 6000.0
TAU_MEAN_LO_LIVE        = 250.0

def recompute_mask(mask_path):
    p = Path(mask_path)
    folder = p.parent
    stem   = p.name.replace('_fit_mask.npy', '')
    m_b = re.search(r'-b(\d+)', folder.name)
    b   = int(m_b.group(1)) if m_b else 1

    ph   = np.loadtxt(folder / f'{stem}_photons.asc').astype(float)
    chi2 = np.loadtxt(folder / f'{stem}_chi.asc')
    a1   = np.loadtxt(folder / f'{stem}_a1.asc')
    a2   = np.loadtxt(folder / f'{stem}_a2.asc')
    t1   = np.loadtxt(folder / f'{stem}_t1.asc')
    t2   = np.loadtxt(folder / f'{stem}_t2.asc')

    k    = 2*b + 1
    binned_ph = uniform_filter(ph, size=k) * (k**2)
    m_ph  = binned_ph >= MIN_PH
    m_chi = np.isfinite(chi2) & (chi2 >= CHI2_LO) & (chi2 <= CHI2_HI)
    m_amp = (a1 >= 0) & (a2 >= 0) & ((a1 + a2) > 0)
    m_tau = (np.isfinite(t1) & (t1 >= TAU1_LO) & (t1 <= TAU1_HI)
           & np.isfinite(t2) & (t2 >= TAU2_LO) & (t2 <= TAU2_HI))
    mask  = m_ph & m_chi & m_amp & m_tau

    if 'live' in stem.lower():
        with np.errstate(invalid='ignore', divide='ignore'):
            asum = a1 + a2
            tm   = np.where(asum > 0, (a1*t1 + a2*t2)/asum, np.nan)
        mask = mask & np.isfinite(tm) & (tm >= TAU_MEAN_LO_LIVE)

    np.save(p, mask)
    return mask.sum(), mask.size

for _, p1, p2 in PAIRS:
    for p in (p1, p2):
        k, n = recompute_mask(p)
        print(f'{Path(p).name[:60]:<60}  {k:>6}/{n} px  ({100*k/n:4.1f}%)')

# %%
# Per-pair pixel-distribution violins for tau_mean and a1/a2.
# Pools only quality-masked pixels; subsamples to 5000 px/file for speed.

records = []
rng = np.random.default_rng(0)
for title, p1, p2 in PAIRS:
    for p in (p1, p2):
        arrs, stem = load_all(p)
        lab = label_of(stem)
        mask = arrs['mask']
        for metric in ('tau_mean', 'a1/a2'):
            vals = arrs[metric][mask]
            vals = vals[np.isfinite(vals)]
            if vals.size > 5000:
                vals = rng.choice(vals, 5000, replace=False)
            records.append(pd.DataFrame({
                'pair': title, 'label': lab, 'metric': metric, 'value': vals
            }))
v = pd.concat(records, ignore_index=True)

# 4 pairs x 2 metrics = 4x2 grid
g = sns.catplot(data=v, x='label', y='value', col='metric', row='pair',
                kind='violin', height=3, aspect=1.4, sharey=False,
                inner='quartile', cut=0)
g.set_titles('{row_name}  |  {col_name}')
g.set_axis_labels('', '')
plt.show()

# %% [markdown]
# ## Step 5: KPC vs BKO dot-plots
#
# Matched subset: Low Pockels, colony_deep, NADH channel (457/475 nm).
# One dot per image (per-image median pixel value).
# Black bars = median +/- 1 SD across images.

# %%
NADH_FILTERS = [457, 475]

subsets = {
    'form 10min': df.loc[(df.fixation_type == 'form')
                         & (df.treatment_duration == '10min')
                         & (df.em_filter_nm.isin(NADH_FILTERS))
                         & (df.PC == 'Low')
                         & (df.annotation == 'colony_deep')],
    'live':       df.loc[(df.fixation_type == 'live')
                         & (df.em_filter_nm.isin(NADH_FILTERS))
                         & (df.PC == 'Low')
                         & (df.annotation == 'colony_deep')],
}

for label, sub in subsets.items():
    print(f'\n{label}: n={len(sub)}')
    print(sub.groupby(['cell_type', 'date']).size())

# %%
METRICS = {
    'tau_mean_median_ps': 'tau_mean (ps)',
    'amp_ratio_median':   'a1 / a2',
}
ORDER   = ['KPCWT', 'BKO']
PALETTE = {'KPCWT': '#3b82f6', 'BKO': '#ef4444'}

fig, axes = plt.subplots(len(METRICS), len(subsets), figsize=(9, 8))

for col, (label, sub) in enumerate(subsets.items()):
    for row, (metric, ylabel) in enumerate(METRICS.items()):
        ax = axes[row][col]
        sns.stripplot(data=sub, x='cell_type', y=metric, order=ORDER,
                      hue='cell_type', palette=PALETTE, legend=False,
                      size=9, alpha=0.75, jitter=0.18, ax=ax)
        # median +/- 1 SD overlay
        for i, ct in enumerate(ORDER):
            vals = sub.loc[sub.cell_type == ct, metric].dropna()
            if vals.empty:
                continue
            med = float(vals.median())
            sd  = float(vals.std())
            ax.errorbar(i, med, yerr=sd, fmt='_', color='black',
                        capsize=14, elinewidth=2.5, markersize=32,
                        markeredgewidth=2.5, zorder=10)
            ax.text(i, med, f'  {med:.1f} +/- {sd:.1f}', fontsize=8,
                    va='center', ha='left')
        # n labels at top
        for i, ct in enumerate(ORDER):
            n = int((sub.cell_type == ct).sum())
            ax.annotate(f'n={n}', xy=(i, 1.0), xycoords=('data', 'axes fraction'),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(ylabel)
        ax.set_title(label, fontsize=12)
        sns.despine(ax=ax)

plt.suptitle('KPCWT vs BKO  |  Low Pockels, colony_deep, NADH (457/475 nm)\n'
             'dots = per-image medians   |   black bars = median +/- SD',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(r'..\results\quickpeek3_dotplots.png', dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 6: Violin + jittered scatter (group-comparison style)
#
# Same matched subset; one panel per (metric, subset).
# Colors from Set2 palette; medians and extrema shown in black.

# %%
_CT_ORDER  = ['KPCWT', 'BKO']
_set2      = plt.cm.Set2(np.linspace(0, 0.8, max(len(_CT_ORDER), 1)))
_CT_COLORS = {ct: _set2[i] for i, ct in enumerate(_CT_ORDER)}

def violin_panel(ax, sub, metric):
    """Build one violin+scatter panel.  Returns (cts_here, groups) for overlay use."""
    cts_here = [ct for ct in _CT_ORDER if (sub.cell_type == ct).sum() >= 2]
    groups   = [sub.loc[sub.cell_type == ct, metric].dropna().values for ct in cts_here]
    colors   = [_CT_COLORS[ct] for ct in cts_here]
    if not groups:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
        return cts_here, groups
    parts = ax.violinplot(groups, positions=range(len(groups)),
                          showmedians=True, showextrema=True)
    for pc, c in zip(parts['bodies'], colors):
        pc.set_facecolor(c); pc.set_alpha(0.65)
    for key in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
        if key in parts:
            parts[key].set_color('k'); parts[key].set_linewidth(1.2)
    rng = np.random.default_rng(0)
    for j, (grp, c) in enumerate(zip(groups, colors)):
        jitter = rng.uniform(-0.07, 0.07, len(grp))
        ax.scatter(j + jitter, grp, s=22, color=c, alpha=0.7, zorder=3, edgecolors='none')
    ax.set_xticks(range(len(cts_here)))
    ax.set_xticklabels(cts_here, fontsize=10)
    return cts_here, groups

fig, axes = plt.subplots(len(METRICS), len(subsets), figsize=(9, 8))
for col, (label, sub) in enumerate(subsets.items()):
    for row, (metric, ylabel) in enumerate(METRICS.items()):
        ax = axes[row][col]
        violin_panel(ax, sub, metric)
        ax.set_ylabel(ylabel)
        n_str = '  '.join(f'{ct}:{(sub.cell_type==ct).sum()}' for ct in _CT_ORDER
                          if (sub.cell_type == ct).sum() > 0)
        ax.set_title(f'{label}\nn={n_str}', fontsize=10)
plt.suptitle('KPCWT vs BKO  |  Low Pockels, colony_deep, NADH (457/475 nm)',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(r'..\results\quickpeek3_violins.png', dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 7: Same violins with closest-to-median marker + example images
#
# For each (subset, cell_type) the row whose (tau_mean, a1/a2) is jointly
# closest to the group median (in z-scored space) is highlighted with a
# black dot and its photons/tau_mean/a1/a2/chi2/mask are shown below.

# %%
def pick_representative(sub, metrics=list(METRICS)):
    out = {}
    for ct in _CT_ORDER:
        s = sub[sub.cell_type == ct].dropna(subset=metrics)
        if s.empty:
            out[ct] = None; continue
        sds  = {m: (s[m].std(ddof=1) or 1.0) for m in metrics}
        meds = {m: s[m].median() for m in metrics}
        d2   = sum(((s[m] - meds[m]) / sds[m])**2 for m in metrics)
        out[ct] = s.loc[d2.idxmin()]
    return out

representatives = {label: pick_representative(sub) for label, sub in subsets.items()}

# -- violin figure with black dots --
fig, axes = plt.subplots(len(METRICS), len(subsets), figsize=(9, 8))
for col, (label, sub) in enumerate(subsets.items()):
    reps = representatives[label]
    for row, (metric, ylabel) in enumerate(METRICS.items()):
        ax = axes[row][col]
        violin_panel(ax, sub, metric)
        for j, ct in enumerate(_CT_ORDER):
            r = reps.get(ct)
            if r is None or pd.isna(r.get(metric)):
                continue
            ax.scatter([j], [r[metric]], s=120, color='black',
                       edgecolors='white', linewidth=1.5, zorder=20)
        ax.set_ylabel(ylabel)
        n_str = '  '.join(f'{ct}:{(sub.cell_type==ct).sum()}' for ct in _CT_ORDER
                          if (sub.cell_type == ct).sum() > 0)
        ax.set_title(f'{label}\nn={n_str}', fontsize=10)
plt.suptitle('Closest-to-median image marked in black',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(r'..\results\quickpeek3_violins_marked.png', dpi=200, bbox_inches='tight')
plt.show()

# -- representative images: 4 rows x 7 cols --
groups_to_show = [(f'{label}  {ct}', representatives[label][ct])
                  for label in subsets for ct in _CT_ORDER
                  if representatives[label].get(ct) is not None]
n_rows = len(groups_to_show)
fig, axes = plt.subplots(n_rows, 7, figsize=(28, 3.5 * n_rows), squeeze=False)
fig.subplots_adjust(hspace=0.6, wspace=0.35, left=0.08)
for ri, (group_label, rep) in enumerate(groups_to_show):
    arrs, stem = load_all(rep['fit_mask_path'])
    show_row(axes[ri], arrs)
    axes[ri][0].set_ylabel(f'{group_label}\nclosest to median',
                            fontsize=11, fontweight='bold',
                            rotation=0, labelpad=90, va='center', ha='right')
plt.suptitle('Representative (closest-to-joint-median) image per group',
             fontsize=12)
plt.savefig(r'..\results\quickpeek3_rep_images.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 8: Experimental variant
#
# - all data shown as dots
# - half-transparent violin for the full distribution
# - thick black bar = interquartile range (Q25..Q75)
# - thin black bar with caps = 95% bootstrap CI of the MEDIAN (rank-based)
# - color-coded median dot (matches its cell-type color so you can match
#   slide-image borders to the dot color later)
# - Mann-Whitney U two-sided p-value (rank-based test, robust to non-normality
#   and pear-shaped distributions)
#
# Note on the previous error bars: matplotlib's default violin "extrema"
# whiskers are min and max of the data, NOT 1 SD or any percentile -- so
# they always cover all points by construction.  We replace them with IQR
# and bootstrap CI of the median.

# %%
from scipy.stats import mannwhitneyu

def violin_full(ax, sub, metric, n_boot=1000):
    cts    = [ct for ct in _CT_ORDER if (sub.cell_type == ct).sum() >= 2]
    groups = [sub.loc[sub.cell_type == ct, metric].dropna().values for ct in cts]
    if not groups:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
        return
    rng = np.random.default_rng(0)
    # Violin (no built-in extrema or median bars)
    parts = ax.violinplot(groups, positions=range(len(groups)),
                          showmedians=False, showextrema=False, widths=0.75)
    for pc, ct in zip(parts['bodies'], cts):
        pc.set_facecolor(_CT_COLORS[ct]); pc.set_alpha(0.35); pc.set_edgecolor('none')
    # All-data scatter
    for j, (grp, ct) in enumerate(zip(groups, cts)):
        jitter = rng.uniform(-0.08, 0.08, len(grp))
        ax.scatter(j + jitter, grp, s=24, color=_CT_COLORS[ct],
                   alpha=0.75, zorder=3, edgecolors='none')
    # IQR + bootstrap 95% CI of median + colored median dot
    for j, (grp, ct) in enumerate(zip(groups, cts)):
        med      = float(np.median(grp))
        q25, q75 = np.percentile(grp, [25, 75])
        # IQR (matched to old box-line weight ~1.2)
        ax.plot([j, j], [q25, q75], color='black', lw=2, zorder=5, solid_capstyle='butt')
        # 95% CI of median by bootstrap
        boot = np.array([np.median(rng.choice(grp, len(grp), replace=True))
                         for _ in range(n_boot)])
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        ax.plot([j, j], [ci_lo, ci_hi], color='black', lw=1.2, zorder=6)
        ax.plot([j-0.10, j+0.10], [ci_lo, ci_lo], color='black', lw=1.2, zorder=6)
        ax.plot([j-0.10, j+0.10], [ci_hi, ci_hi], color='black', lw=1.2, zorder=6)
        # Median dot (color-coded so slide image borders can match)
        ax.scatter([j], [med], s=160, color=_CT_COLORS[ct],
                   edgecolors='black', linewidth=2, zorder=10)
    # Mann-Whitney U two-sided
    if len(groups) == 2 and len(groups[0]) >= 2 and len(groups[1]) >= 2:
        u, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
        sig  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.annotate(f'MW U  p={p:.2g}  {sig}',
                    xy=(0.5, 0.97), xycoords='axes fraction',
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.85))
    ax.set_xticks(range(len(cts)))
    ax.set_xticklabels(cts, fontsize=10)
    n_str = '  '.join(f'{ct}:{(sub.cell_type==ct).sum()}' for ct in _CT_ORDER
                      if (sub.cell_type == ct).sum() > 0)
    ax.set_title(f'n={n_str}', fontsize=9)
    return cts, groups

fig, axes = plt.subplots(len(METRICS), len(subsets), figsize=(10, 9))
for col, (label, sub) in enumerate(subsets.items()):
    for row, (metric, ylabel) in enumerate(METRICS.items()):
        ax = axes[row][col]
        violin_full(ax, sub, metric)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{label}\n{ax.get_title()}', fontsize=10)
plt.suptitle('Dots (all data) + violin + IQR + 95% bootstrap-CI of median '
             '  |  Mann-Whitney U rank test',
             fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(r'..\results\quickpeek3_violins_v2.png', dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 9: Live data session-split sensitivity check
#
# The KPC live data looks bimodal because it pools 5/1 and 5/8 acquisitions,
# which used different microscope settings.  Three subsets are plotted side
# by side: combined, then KPC restricted to 5/1, then KPC restricted to 5/8.
# In all three, BKO is whatever passes the matched-conditions filter
# (currently 5/17).  This is the figure to use to explain inter-session
# variability to collaborators.

# %%
live_base = subsets['live']
live_subsets = {
    'live (5/1 + 5/8)':  live_base,
    'live (KPC 5/1)':    live_base[(live_base.cell_type == 'BKO')
                                   | (live_base.date == pd.Timestamp('2026-05-01'))],
    'live (KPC 5/8)':    live_base[(live_base.cell_type == 'BKO')
                                   | (live_base.date == pd.Timestamp('2026-05-08'))],
}
for label, sub in live_subsets.items():
    print(f'{label}: n={len(sub)}')
    print(sub.groupby(['cell_type', 'date']).size())
    print()

# %%
fig, axes = plt.subplots(len(METRICS), len(live_subsets), figsize=(14, 9))
for col, (label, sub) in enumerate(live_subsets.items()):
    for row, (metric, ylabel) in enumerate(METRICS.items()):
        ax = axes[row][col]
        violin_full(ax, sub, metric)
        ax.set_ylabel(ylabel if col == 0 else '')
        ax.set_title(f'{label}\n{ax.get_title()}', fontsize=10)
plt.suptitle('Live data session-split  |  BKO held fixed (5/17), '
             'KPC restricted as labelled',
             fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(r'..\results\quickpeek3_live_split.png', dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 10: Phasor diagrams
#
# - **Group phasor**: one dot per image at the photon-weighted mean phasor
#   (G_cal_wmean, S_cal_wmean from `sdt_phasor_summary.csv`).
# - **Pixel phasor**: 2D histogram of per-pixel calibrated G, S pooled across
#   all images in the subset, filtered by Phase E quality mask.  Recomputed
#   on the fly from each .sdt file using the calibration in
#   `sdt_metadata_cal.csv` (phasor_cal_phase_rad, phasor_cal_mod).
#
# Universal semicircle drawn from (0,0) to (1,0) through (0.5, 0.5);
# single-exponential lifetimes lie on it.

# %%
# Pull G_cal_wmean / S_cal_wmean and calibration / filepath into df.
if 'G_cal_wmean' not in df.columns:
    phasor_summary = pd.read_csv(r'..\results\sdt_phasor_summary.csv')
    df = df.merge(phasor_summary[['filename', 'G_cal_wmean', 'S_cal_wmean']],
                  on='filename', how='left')
if 'phasor_cal_phase_rad' not in df.columns:
    df = df.merge(sdt[['filename', 'phasor_cal_phase_rad', 'phasor_cal_mod']],
                  on='filename', how='left')
if 'filepath' not in df.columns:
    fp_map = pd.read_csv(r'..\results\filepath_map.csv')
    df = df.merge(fp_map[['filename', 'filepath']], on='filename', how='left')

# Refresh subsets so they pick up the new columns
def _filter_form_live():
    fl = df.loc[(df.fixation_type == 'form')
                & (df.treatment_duration == '10min')
                & (df.em_filter_nm.isin(NADH_FILTERS))
                & (df.PC == 'Low')
                & (df.annotation == 'colony_deep')]
    lv = df.loc[(df.fixation_type == 'live')
                & (df.em_filter_nm.isin(NADH_FILTERS))
                & (df.PC == 'Low')
                & (df.annotation == 'colony_deep')]
    return fl, lv

_form, _live = _filter_form_live()
phasor_subsets = {
    'form 10min':       _form,
    'live (5/1+5/8)':   _live,
    'live (KPC 5/1)':   _live[(_live.cell_type == 'BKO')
                              | (_live.date == pd.Timestamp('2026-05-01'))],
    'live (KPC 5/8)':   _live[(_live.cell_type == 'BKO')
                              | (_live.date == pd.Timestamp('2026-05-08'))],
}

def draw_semicircle(ax):
    theta = np.linspace(0, np.pi, 200)
    ax.plot(0.5 + 0.5*np.cos(theta), 0.5*np.sin(theta), color='gray', lw=1)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 0.6)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('G'); ax.set_ylabel('S')

# %% Group phasor (one point per image)
fig, axes = plt.subplots(1, len(phasor_subsets),
                         figsize=(5.5*len(phasor_subsets), 5.5), squeeze=False)
for col, (label, sub) in enumerate(phasor_subsets.items()):
    ax = axes[0][col]
    draw_semicircle(ax)
    for ct in _CT_ORDER:
        s = sub[sub.cell_type == ct].dropna(subset=['G_cal_wmean', 'S_cal_wmean'])
        ax.scatter(s.G_cal_wmean, s.S_cal_wmean, color=_CT_COLORS[ct],
                   s=55, alpha=0.8, edgecolors='black', linewidth=0.5,
                   label=f'{ct} (n={len(s)})', zorder=4)
    ax.set_title(label, fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
plt.suptitle('Group phasor  |  one point per image at photon-weighted (G_cal, S_cal)',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(r'..\results\quickpeek3_phasor_group.png', dpi=200, bbox_inches='tight')
plt.show()

# %% Pixel phasor (slow: loads .sdt files; results cached per filename)
from sdtfile import SdtFile
from matplotlib.colors import LogNorm

REP_RATE_HZ = 80e6
OMEGA       = 2.0 * np.pi * REP_RATE_HZ

def _pixel_phasor(filepath, phase_cal, mod_cal):
    sdt_obj  = SdtFile(str(filepath))
    decay    = sdt_obj.data[0].astype(float)
    times_ns = sdt_obj.times[0].astype(float) * 1e9
    omega_ns = OMEGA * 1e-9
    photons  = decay.sum(axis=2)
    safe_n   = np.where(photons > 0, photons, 1.0)
    cos_t    = np.cos(omega_ns * times_ns)
    sin_t    = np.sin(omega_ns * times_ns)
    G_raw    = np.tensordot(decay, cos_t, axes=[[2], [0]]) / safe_n
    S_raw    = np.tensordot(decay, sin_t, axes=[[2], [0]]) / safe_n
    cos_phi  = np.cos(phase_cal)
    sin_phi  = np.sin(phase_cal)
    G_cal    = mod_cal * (G_raw * cos_phi - S_raw * sin_phi)
    S_cal    = mod_cal * (G_raw * sin_phi + S_raw * cos_phi)
    return G_cal, S_cal, photons

_pixel_cache = {}
def _get_masked_pixels(row):
    fn = row['filename']
    if fn in _pixel_cache:
        return _pixel_cache[fn]
    fp = row.get('filepath')
    if pd.isna(fp) or pd.isna(row.get('phasor_cal_phase_rad')):
        _pixel_cache[fn] = (np.array([]), np.array([])); return _pixel_cache[fn]
    try:
        G, S, ph = _pixel_phasor(fp, row['phasor_cal_phase_rad'], row['phasor_cal_mod'])
        mp = row.get('fit_mask_path')
        if isinstance(mp, str) and Path(mp).exists():
            mask = np.load(mp)
        else:
            mask = ph >= 100   # fallback if no Phase E mask available
    except Exception as e:
        print(f'  skip {fn[-50:]}: {e}')
        _pixel_cache[fn] = (np.array([]), np.array([])); return _pixel_cache[fn]
    _pixel_cache[fn] = (G[mask], S[mask])
    return _pixel_cache[fn]

def pool_pixels(sub):
    Gs, Ss = [], []
    for _, row in sub.iterrows():
        G, S = _get_masked_pixels(row)
        if G.size: Gs.append(G); Ss.append(S)
    return (np.concatenate(Gs), np.concatenate(Ss)) if Gs else (np.array([]), np.array([]))

# 2 rows (KPCWT, BKO) x 4 cols (subsets)
fig, axes = plt.subplots(2, len(phasor_subsets),
                         figsize=(5.5*len(phasor_subsets), 10), squeeze=False)
for col, (label, sub) in enumerate(phasor_subsets.items()):
    print(f'\n{label}:')
    for row_idx, ct in enumerate(_CT_ORDER):
        ax = axes[row_idx][col]
        draw_semicircle(ax)
        s = sub[sub.cell_type == ct]
        print(f'  {ct}: {len(s)} files...')
        G, S = pool_pixels(s)
        if G.size:
            h = ax.hist2d(G, S, bins=120, range=[[0, 1], [0, 0.6]],
                          cmap='hot', norm=LogNorm(vmin=1), cmin=1)
            plt.colorbar(h[3], ax=ax, fraction=0.04, pad=0.04)
        ax.set_title(f'{label}   {ct}  (n={len(s)} files, {G.size:,} px)',
                     fontsize=9)
plt.suptitle('Per-pixel phasor (Phase-E mask, pooled across images)',
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(r'..\results\quickpeek3_phasor_pixel.png', dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 11: Sensitivity analysis - Pockels cell
#
# Forest plot of (KPCWT - BKO) median difference, stratified by Pockels.
# Each row gives the effect within one Pockels stratum with:
#   - point estimate (median diff)
#   - 95% bootstrap CI on the median diff
#   - n per group, rank-biserial r, Mann-Whitney U p-value
#
# Interpretation:
#   - CI crosses 0      -> non-significant in that stratum
#   - direction flips   -> effect is confounded with Pockels (not real biology)
#   - consistent across strata with narrow CI  -> robust
# CI width vs effect size answers the "are we overfitting?" question
# quantitatively: if half-width >> effect, the slice is too small / too noisy.

# %%
from scipy.stats import mannwhitneyu

def effect_stats(sub, metric, n_boot=2000, seed=0):
    rng  = np.random.default_rng(seed)
    kpc  = sub.loc[sub.cell_type == 'KPCWT', metric].dropna().to_numpy()
    bko  = sub.loc[sub.cell_type == 'BKO',   metric].dropna().to_numpy()
    if len(kpc) < 2 or len(bko) < 2:
        return None
    obs  = float(np.median(kpc) - np.median(bko))
    boot = np.array([
        np.median(rng.choice(kpc, len(kpc), replace=True)) -
        np.median(rng.choice(bko, len(bko), replace=True))
        for _ in range(n_boot)
    ])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    u, p = mannwhitneyu(kpc, bko, alternative='two-sided')
    r    = 2 * u / (len(kpc) * len(bko)) - 1   # rank-biserial; +ve = KPC>BKO
    return dict(diff=obs, ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                p=float(p), r=float(r), n_kpc=int(len(kpc)), n_bko=int(len(bko)))

df['pockels_stratum'] = pd.cut(
    df.pockels, bins=[0, 0.2, 0.25, 0.3, 0.5, 1.0],
    labels=['<=0.20', '0.21-0.25', '0.26-0.30', '0.31-0.50', '>0.50'],
)

def strata_for(fixation, treatment=None):
    base = ((df.fixation_type == fixation)
            & df.em_filter_nm.isin(NADH_FILTERS)
            & (df.annotation == 'colony_deep'))
    if treatment is not None:
        base &= (df.treatment_duration == treatment)
    return [(str(s), df.loc[base & (df.pockels_stratum == s)])
            for s in df.pockels_stratum.cat.categories]

EXPERIMENTS = [('form 10min', strata_for('form', '10min')),
               ('live',       strata_for('live'))]

fig, axes = plt.subplots(len(METRICS), len(EXPERIMENTS),
                         figsize=(9 * len(EXPERIMENTS), 4.5 * len(METRICS)),
                         squeeze=False)
for col, (exp_label, strata) in enumerate(EXPERIMENTS):
    for row, (metric, ylabel) in enumerate(METRICS.items()):
        ax  = axes[row][col]
        rows_data = [(lbl, effect_stats(sub, metric)) for lbl, sub in strata]
        for i, (lbl, eff) in enumerate(rows_data):
            if eff is None:
                ax.text(0.02, i, '(n < 2 per group)', va='center',
                        transform=ax.get_yaxis_transform(),
                        fontsize=8, color='gray')
                continue
            color = 'black' if (eff['ci_lo'] * eff['ci_hi'] > 0) else 'gray'
            ax.errorbar([eff['diff']], [i],
                        xerr=[[eff['diff']-eff['ci_lo']], [eff['ci_hi']-eff['diff']]],
                        fmt='o', color=color, capsize=4, lw=1.5, markersize=8)
            sig = ' *' if eff['p'] < 0.05 else ''
            ax.text(1.02, i,
                    f"n=({eff['n_kpc']},{eff['n_bko']})  "
                    f"r={eff['r']:+.2f}  p={eff['p']:.3f}{sig}",
                    transform=ax.get_yaxis_transform(),
                    fontsize=9, va='center', family='monospace')
        ax.axvline(0, color='gray', lw=1, linestyle='--')
        ax.set_yticks(range(len(rows_data)))
        ax.set_yticklabels([r[0] for r in rows_data])
        ax.invert_yaxis()
        ax.set_xlabel(f'KPCWT - BKO median diff: {ylabel}')
        ax.set_title(f'{exp_label}  |  {ylabel}', fontsize=10)
plt.suptitle('Pockels sensitivity  |  forest plot, KPC - BKO median diff with '
             '95% bootstrap CI', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(r'..\results\quickpeek3_sensitivity_pockels.png',
            dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Permutation test on the primary stratum** (overfitting check):
# shuffle cell_type labels 2000x within the matched subset, recompute the
# median difference each time.  Where the observed effect falls in the
# permutation null distribution is the empirical p-value.

# %%
def perm_test(sub, metric, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    kpc = sub.loc[sub.cell_type == 'KPCWT', metric].dropna().to_numpy()
    bko = sub.loc[sub.cell_type == 'BKO',   metric].dropna().to_numpy()
    if len(kpc) < 2 or len(bko) < 2:
        return None
    obs    = float(np.median(kpc) - np.median(bko))
    pooled = np.concatenate([kpc, bko])
    n_k    = len(kpc)
    null   = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        null[i] = np.median(pooled[:n_k]) - np.median(pooled[n_k:])
    p_emp  = float((np.abs(null) >= abs(obs)).mean())
    return obs, null, p_emp

primary = {
    'form 10min': subsets['form 10min'],
    'live':       subsets['live'],
}

fig, axes = plt.subplots(len(METRICS), len(primary),
                         figsize=(5 * len(primary), 3.5 * len(METRICS)),
                         squeeze=False)
for col, (label, sub) in enumerate(primary.items()):
    for row, (metric, ylabel) in enumerate(METRICS.items()):
        ax  = axes[row][col]
        res = perm_test(sub, metric)
        if res is None:
            ax.text(0.5, 0.5, '(n < 2)', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        obs, null, p_emp = res
        ax.hist(null, bins=40, color='lightgray', edgecolor='gray')
        ax.axvline(obs, color='red', lw=2.5, label=f'obs = {obs:+.3f}')
        ax.axvline(0,   color='black', lw=0.8, linestyle=':')
        ax.set_title(f'{label}  |  {ylabel}\np_perm = {p_emp:.4f}', fontsize=10)
        ax.set_xlabel('median(KPCWT) - median(BKO) under null')
        ax.set_ylabel('count')
        ax.legend(fontsize=9)
plt.suptitle('Permutation test on primary subset  |  null = shuffled labels',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(r'..\results\quickpeek3_permutation.png', dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 12: Live-session sensitivity (drop 5/1 KPC and 5/21 BKO)
#
# Justification for the restriction: Pockels is a significant factor for live
# tau_mean and for form 10min tau_mean and a1/a2 (Step 11 forest plots).
# - 5/1 KPC was acquired at Pockels=0.1
# - 5/21 BKO was acquired at Pockels=0.25
# - 5/8 KPC and 5/17 BKO were both at Pockels=0.2  -- best match.
#
# Forest of (KPCWT - BKO) median diff under three restrictions:
#   primary       = full Low-PC matched subset
#   drop 5/21 BKO = keep both KPC sessions, drop 5/21 BKO
#   only 5/8+5/17 = keep only Pockels=0.2 sessions

# %%
live_base = df.loc[(df.fixation_type == 'live')
                   & df.em_filter_nm.isin(NADH_FILTERS)
                   & (df.PC == 'Low')
                   & (df.annotation == 'colony_deep')]

live_restrictions = {
    'primary (all Low PC)': live_base,
    'drop 5/21 BKO':        live_base.loc[live_base.date != pd.Timestamp('2026-05-21')],
    'only 5/8 + 5/17':      live_base.loc[live_base.date.isin(
                                [pd.Timestamp('2026-05-08'),
                                 pd.Timestamp('2026-05-17')])],
}

fig, axes = plt.subplots(len(METRICS), 1,
                         figsize=(10, 3.5 * len(METRICS)), squeeze=False)
for row, (metric, ylabel) in enumerate(METRICS.items()):
    ax = axes[row][0]
    items = []
    for lbl, sub in live_restrictions.items():
        items.append((lbl, effect_stats(sub, metric)))
    for i, (lbl, eff) in enumerate(items):
        if eff is None:
            ax.text(0.02, i, '(n < 2)', va='center',
                    transform=ax.get_yaxis_transform(),
                    fontsize=9, color='gray')
            continue
        color = 'black' if (eff['ci_lo'] * eff['ci_hi'] > 0) else 'gray'
        ax.errorbar([eff['diff']], [i],
                    xerr=[[eff['diff']-eff['ci_lo']], [eff['ci_hi']-eff['diff']]],
                    fmt='o', color=color, capsize=4, lw=1.6, markersize=9)
        sig = ' *' if eff['p'] < 0.05 else ''
        ax.text(1.02, i,
                f"n=({eff['n_kpc']},{eff['n_bko']})  "
                f"r={eff['r']:+.2f}  p={eff['p']:.3f}{sig}",
                transform=ax.get_yaxis_transform(),
                fontsize=9, va='center', family='monospace')
    ax.axvline(0, color='gray', lw=1, linestyle='--')
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([r[0] for r in items])
    ax.invert_yaxis()
    ax.set_xlabel(f'KPCWT - BKO median diff: {ylabel}')
    ax.set_title(ylabel, fontsize=10)
plt.suptitle('Live: session-restriction sensitivity  |  matched-Pockels subset',
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(r'..\results\quickpeek3_live_session_sensitivity.png',
            dpi=200, bbox_inches='tight')
plt.show()

# %%
