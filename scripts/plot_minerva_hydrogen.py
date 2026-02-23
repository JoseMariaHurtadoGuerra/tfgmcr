# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 21:58:18 2026

@author: User
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Paths
# ---------------------------
PROJECT_ROOT = r"C:\Users\User\Documents\tfgmcr"
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "minerva_hydrogen")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

XSEC_PATH = os.path.join(RAW_DIR, "hydrogen_xsec.csv")
COV_TOT_PATH = os.path.join(RAW_DIR, "cov_tot.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------
# Load cross section table
# ---------------------------
xsec = pd.read_csv(XSEC_PATH)

required = {"Q2center", "Q2low", "Q2High", "xsec"}
missing = required - set(xsec.columns)
if missing:
    raise RuntimeError(f"Faltan columnas en hydrogen_xsec.csv: {missing}. "
                       f"Columnas disponibles: {list(xsec.columns)}")

Q2c = xsec["Q2center"].to_numpy(dtype=float)
Q2lo = xsec["Q2low"].to_numpy(dtype=float)
Q2hi = xsec["Q2High"].to_numpy(dtype=float)
y = xsec["xsec"].to_numpy(dtype=float)
n = len(y)

# Optional columns (may exist)
stat = xsec["stat"].to_numpy(dtype=float) if "stat" in xsec.columns else None
sys  = xsec["sys"].to_numpy(dtype=float) if "sys"  in xsec.columns else None
yerr_quad = np.sqrt(stat**2 + sys**2) if (stat is not None and sys is not None) else None

# ---------------------------
# Load covariance (IMPORTANT: header=None)
# ---------------------------
cov = pd.read_csv(COV_TOT_PATH, header=None)
cov_vals = cov.to_numpy(dtype=float)

if cov_vals.shape != (n, n):
    raise RuntimeError(
        f"La covarianza no cuadra: n bins={n} pero cov shape={cov_vals.shape}. "
        "Abre cov_tot.csv y revisa si tiene cabeceras/filas extra."
    )

yerr_tot = np.sqrt(np.diag(cov_vals))

# ---------------------------
# Save processed CSV
# ---------------------------
out = pd.DataFrame({
    "Q2center": Q2c,
    "Q2low": Q2lo,
    "Q2high": Q2hi,
    "xsec": y,
    "xsec_err_tot": yerr_tot,
})

if stat is not None:
    out["xsec_err_stat"] = stat
if sys is not None:
    out["xsec_err_sys"] = sys

out_path = os.path.join(PROCESSED_DIR, "minerva_hydrogen_xsec_processed.csv")
out.to_csv(out_path, index=False)

# ---------------------------
# Plot (PDF vector)
# ---------------------------
# For log-scale plotting we must remove non-positive points
mask = (y > 0) & (yerr_tot > 0)
Q2c_p = Q2c[mask]
y_p = y[mask]
yerr_tot_p = yerr_tot[mask]

# horizontal error bars from bin widths
xerr_low = Q2c_p - Q2lo[mask]
xerr_high = Q2hi[mask] - Q2c_p
xerr = np.vstack([xerr_low, xerr_high])

plt.figure()

plt.errorbar(
    Q2c_p, y_p,
    xerr=xerr,
    yerr=yerr_tot_p,
    fmt="o", capsize=3,
    label="MINERvA (tot)"
)

# Optional check: stat ⊕ sys (diagonal-only)
if yerr_quad is not None:
    yerr_quad_p = yerr_quad[mask]
    plt.errorbar(
        Q2c_p, y_p,
        xerr=xerr,
        yerr=yerr_quad_p,
        fmt="none", capsize=3,
        label=r"stat $\oplus$ sys (diag)"
    )

plt.xlabel(r"$Q^2\ \mathrm{[GeV^2]}$")
plt.ylabel(r"$\mathrm{d}\sigma/\mathrm{d}Q^2\ [10^{-38}\ cm^2/GeV^2/H]$")

plt.yscale("log")
plt.grid(True, which="both", alpha=0.3)
plt.legend()

# Set y-limits nicely (avoid huge empty space)
ymin = np.min(y_p - yerr_tot_p[y_p > 0]) if np.any(y_p > 0) else np.min(y_p)
ymin = max(ymin, 1e-4)
plt.ylim(ymin, None)

plt.tight_layout()

fig_path = os.path.join(FIG_DIR, "minerva_hydrogen_xsec.pdf")
plt.savefig(fig_path)
plt.show()

print("\nGuardado:")
print(" -", out_path)
print(" -", fig_path)
print(f"Usados {mask.sum()}/{n} bins (se filtran xsec<=0 para escala log).")