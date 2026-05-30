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

# %%
import pandas as pd

# %%
pwd()

# %%
df = pd.read_csv(r"..\results\fit_analysis_summary.csv")

# %%
df.head()

# %%
#df['fix_time']=
df['date'] =  pd.to_datetime(df.session_root.str.split('_').str[0], format='%Y%m%d')

# %%
df['fix_time']=[int(k.lower().split('form')[1][:2]) if k.lower().find('form')>-1 else 0 for k in df.filename.values ]


# %%
df.columns

# %%
df['poc'] = df.filename.str.split('_poc').str[1].str.split('_').str[0]
df['pockelcell'] = df.poc.str.replace('p','.').astype(float)

# %%
df['PC'] = ['High' if k>0.3 else 'Low' for k in df.pockelcell.values]
df.PC.value_counts()

# %% [markdown]
# # data describe

# %%
df.groupby(['fixation_type', 'em_filter_nm', 'date','cell_type','fix_time', 'PC']).size()

# %%

# %%
df.session_root.unique(), df.cell_type.unique()

# %%
import seaborn as sns

# %%
sns.catplot(x='session_root',y='amp_ratio_mean', data=df.loc[(df.em_filter_nm==457)], hue='cell_type',col='fixation_type')


# %%
sns.catplot(x='session_root',y='amp_ratio_mean', data=df.loc[(df.em_filter_nm==457) & (df.PC == "Low") ], hue='cell_type',col='fixation_type')


# %%
df.loc[df.em_filter_nm==457].shape

# %% [markdown]
# ## a1/a2 median

# %%
sns.catplot(x='session_root',y='amp_ratio_median', data=df.loc[df.em_filter_nm==457], hue='cell_type',col='fixation_type')


# %%
df.columns

# %%
sns.catplot(x='date',y='tau_mean_median_ps', data=df.loc[df.em_filter_nm==457], hue='cell_type',col='fixation_type')


# %% [markdown]
# ## taumean

# %%
sns.catplot(x='date',y='tau_mean_mean_ps', data=df.loc[df.em_filter_nm==457], hue='cell_type',col='fixation_type',kind='box')


# %% [markdown]
# ## two pockel cell data filter

# %%



# %%

# %%
sns.catplot(x='date',y='tau_mean_mean_ps', data=df.loc[df.em_filter_nm==457], hue='cell_type',col='fixation_type',kind='box',row='PC')


# %%
sns.catplot(x='date',y='tau_mean_median_ps', data=df.loc[(df.em_filter_nm==457)&(df.PC=='Low')], hue='cell_type',col='fixation_type')


# %%
sns.catplot(x='date',y='amp_ratio_mean', data=df.loc[(df.em_filter_nm==457)&(df.PC=='Low')], hue='cell_type',col='fixation_type')


# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# # reading asc

# %%
filenames = df.loc[df.em_filter_nm==457]['filename'].unique()
len(filenames)

# %%
# %%time
from pathlib import Path
sdtfiles = list(Path(r"E:\18_RK_Circadian\data\raw").rglob('*.sdt'))
len(sdtfiles)

# %%
for filename in filenames:
    #print(filename)
    corr_file_path = [k for k in sdtfiles if k.name == filename]
    
    assert len(corr_file_path)==1, f"Expected exactly one match for {filename}, found {len(corr_file_path)}"


# %%

# %%
fn = Path(filename)
[k for k in list(corr_file_path[0].parent.rglob('*.asc')) if (k.name.startswith(fn.stem))&(k.name.endswith('photons.asc'))]

# %%
import numpy as np

# %%
for filename in filenames:
    #print(filename)
    corr_file_path = [k for k in sdtfiles if k.name == filename]
    assert len(corr_file_path)==1, f"Expected exactly one match for {filename}, found {len(corr_file_path)}"
    fn = Path(filename)
    asc_ph_files = [k for k in list(corr_file_path[0].parent.rglob('*.asc')) if (k.name.startswith(fn.stem))&(k.name.endswith('photons.asc'))]
    szfiles = [k for k in asc_ph_files if k.parts[-2].find('-sz-')>-1]
    for szfile in szfiles:
        shiftfile = str(szfile).replace('photons.asc','shift.asc') 
        shift_data = np.loadtxt(shiftfile)
        if shift_data.sum()>0:
            print(szfile.name,  shift_data.sum())
        #break
    #break

# %% [markdown]
# ### SPC IMAGE 8.9.85 keeps the shift to zero on MLE if you set the checkbox as fixed

# %%
fn_dict={}
for filename in filenames:
    #print(filename)
    corr_file_path = [k for k in sdtfiles if k.name == filename]
    assert len(corr_file_path)==1, f"Expected exactly one match for {filename}, found {len(corr_file_path)}"
    fn = Path(filename)
    asc_ph_files = [k for k in list(corr_file_path[0].parent.rglob('*.asc')) if (k.name.startswith(fn.stem))&(k.name.endswith('photons.asc'))]
    szfiles = [k for k in asc_ph_files if k.parts[-2].find('fitet-sz-c2')>-1]
    fn_dict[filename]=szfiles[-1]


# %%
fn_dict[filename].name

# %%
from matplotlib.pyplot import *

# %%
from skimage.filters import threshold_local

# %%

# %%
data_dict = {}
for filename in fn_dict:
    #print(filename, fn_dict[filename].parent)
    ph_file = np.loadtxt(fn_dict[filename])
    color_coded_file = Path(str(fn_dict[filename]).replace('photons.asc','color coded value.asc'))
    if color_coded_file.exists():
        taumean_file = np.loadtxt(color_coded_file)
    else:
        taumean_file = np.array([np.nan])
    a1_file = Path(str(fn_dict[filename]).replace('photons.asc','a1.asc')) 
    if a1_file.exists():
        a1_file = np.loadtxt(a1_file)
    else:
        a1_file = np.array([np.nan])
    a2_file = Path(str(fn_dict[filename]).replace('photons.asc', 'a2.asc'))
    if a2_file.exists():
        a2_file = np.loadtxt(a2_file)
    else:
        a2_file = np.array([np.nan])
    #a1a2 = a1_file/a2_file
    chi_file = Path(str(fn_dict[filename]).replace('photons.asc', 'chi.asc'))
    if chi_file.exists():
        chi_file = np.loadtxt(chi_file)
    else:
        chi_file = np.array([np.nan])
    ma = ph_file > ph_file.mean()*1.2
    #mask = threshold_local(ph_file, block_size=11, offset=8)
    #imshow(ma)
    data_dict[filename] = {
        'filepath':fn_dict[filename],
        'ph':ph_file[ma].mean(), 
        'taumean':taumean_file[ma].mean(),
        'a1':a1_file[ma].mean(),
        'a2':a2_file[ma].mean(),
        'chi':chi_file[ma].mean()}
    
    make_fig()
    output = Path(r'E:\18_RK_Circadian\data\processed_temp\preview') / (fn_dict[filename].stem + '_preview.png')
    savefig(output)
    close()
    
    #break


# %%
#ph_file[ma].mean(), ph_file[~ma].mean(), a1_file[ma].mean(), a1_file[~ma].mean()

# %%
def make_fig():
    figure(figsize=(25,3))
    subplot(1,6,1)
    imshow(ph_file)
    title(f'{ph_file[ma].mean():.1f} vs {ph_file[~ma].mean():.1f}')
    colorbar()
    subplot(1,6,2) 
    imshow(taumean_file)
    title(f'{taumean_file[ma].mean():.1f} vs {taumean_file[~ma].mean():.1f}')
    colorbar()
    subplot(1,6 ,3)  
    imshow(a1_file)
    title(f'{a1_file[ma].mean():.1f} vs {a1_file[~ma].mean():.1f}')
    colorbar()
    subplot(1,6,4)  
    imshow(a2_file)
    title(f'{a2_file[ma].mean():.1f} vs {a2_file[~ma].mean():.1f}')
    colorbar()
    subplot(1,6,5)
    imshow(chi_file)
    title(f'{chi_file[ma].mean():.1f} vs {chi_file[~ma].mean():.1f}')
    colorbar()
    subplot(1,6,6)
    imshow(ma)
    title(f'Mask: {ma.sum()} pixels')
    colorbar()



# %%
df1 = pd.DataFrame.from_dict(data_dict, orient='index')

# %%
df1.reset_index(inplace=True)

# %%
df1.columns

# %%
df1.columns = ['filename', 'filepath', 'ph_mean', 'taumean_mean', 'a1_mean', 'a2_mean', 'chi_mean']

# %%
df1.head()

# %%
df1['datex'] = [k.parts[4].split('_')[0] for k in df1.filepath.values]
df1.datex.value_counts()

# %%
df1['fixation'] = ['L' if k.lower().find('live')>-1 else 'F' for k in df1.filename.values] 
df1.fixation.value_counts()

# %%
df1['fixation'] = ['L' if k.lower().find('live')>-1 else 'F' for k in df1.filename.values] 
df1.fixation.value_counts()

# %%
df1['celltype'] = [k.split('_')[1][0] for k in df1.filename.values] 
df1.celltype.value_counts()

# %%
df1.head()

# %%
sns.catplot(x='datex',y='taumean_mean', data=df1, hue='celltype',col='fixation',kind='box')

# %%
df1['a1bya2'] = df1.a1_mean/df1.a2_mean

# %%
sns.catplot(x='datex',y='a1bya2', data=df1, hue='celltype',col='fixation',kind='strip')

# %%
## 8-cell fits

# %%
Cells

# %%
