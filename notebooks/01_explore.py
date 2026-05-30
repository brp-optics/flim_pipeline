# %% [markdown]
# ## IRF inspection
# Load and inspect the instrument response function before fitting.

# %%
import sdtfile
import matplotlib.pyplot as plt
import numpy as np
import os


# %%
# Inline images
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")

sdt = sdtfile.SdtFile("data/raw/irf_reference.sdt")
irf = sdt.data[0].sum(axis=(0, 1))  # collapse spatial dims

# %%
plt.plot(sdt.times[0] * 1e9, irf)
plt.xlabel("time (ns)")
plt.ylabel("photon counts")
plt.title("IRF")
plt.show()
# %%

# %matplotlib inline
# %%
print(os.getcwd())


# %%

from src.preprocess import estimate_irf_width
fwhm = estimate_irf_width(irf, sdt.times[0])
print(f"IRF FWHM: {fwhm*1e12:.1f} ps")

