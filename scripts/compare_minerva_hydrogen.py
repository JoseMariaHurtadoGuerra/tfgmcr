# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:16:47 2026

@author: User
"""

# scripts/compare_minerva_hydrogen.py
from __future__ import annotations
print("RUNNING:", __file__)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Robust project paths
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from minerva.flux_folding import flux_folded_binned_xsec
from ccqe_hydrogen_xsec import dsigma_dQ2_numubar_p


# -----------------------
# Column helpers
# -----------------------
def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = list(df.columns)
    strip_map = {c.strip(): c for c in cols}

    # exact
    for cand in candidates:
        if cand in strip_map:
            return strip_map[cand]

    # exact lower
    lower_exact = {c.strip().lower(): c for c in cols}
    for cand in candidates:
        cl = cand.strip().lower()
        if cl in lower_exact:
            return lower_exact[cl]

    # substring
    cols_low = [(c, c.strip().lower()) for c in cols]
    for cand in candidates:
        cl = cand.strip().lower()
        for orig, low in cols_low:
            if cl in low:
                return orig

    raise KeyError(f"No encuentro columnas {candidates}. Tengo: {list(df.columns)}")


def pick_flux_csv(raw_dir: Path) -> Path:
    """
    Picks a flux CSV by checking it has an Energy column and a column that starts with 'flux('.
    """
    for p in sorted(raw_dir.glob("*.csv")):
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        cols = [c.strip() for c in df.columns]
        has_energy = any("energy" in c.lower() for c in cols)
        has_flux = any(c.lower().startswith("flux(") for c in cols)
        if has_energy and has_flux:
            return p
    raise FileNotFoundError(f"No encuentro un CSV de flujo válido en {raw_dir}.")


def pick_flux_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.strip().lower().startswith("flux("):
            return c
    return find_col(df, ["flux", "phi"])


def read_cov_matrix(path: Path, n: int) -> np.ndarray:
    V = pd.read_csv(path, header=None).to_numpy(dtype=float)

    # common cases: extra index column/row
    if V.shape == (n, n + 1):
        V = V[:, 1:]
    if V.shape == (n + 1, n + 1):
        V = V[1:, 1:]
    if V.shape != (n, n):
        raise ValueError(f"Covarianza con forma {V.shape}, esperaba {(n, n)}. Revisa {path.name}.")
    return V


# -----------------------
# Model wrapper (already MINERvA units)
# -----------------------
def dsigma_dQ2_model(Ev: float, Q2: float, params: dict) -> float:
    """
    Returns dσ/dQ^2 in [1e-38 cm^2/GeV^2] for \bar{nu}_mu p -> mu^+ n.
    """
    MA = float(params.get("MA", 1.00))
    MV2 = float(params.get("MV2", 0.71))
    return float(dsigma_dQ2_numubar_p(Ev, Q2, MA=MA, MV2=MV2))


# -----------------------
# Main
# -----------------------
def main():
    raw_dir = PROJECT_ROOT / "data" / "raw" / "minerva_hydrogen"
    proc_dir = PROJECT_ROOT / "data" / "processed" / "minerva_hydrogen"
    fig_dir = PROJECT_ROOT / "results" / "figures"
    proc_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- DATA (bins + xsec) ----
    xsec_path = raw_dir / "hydrogen_xsec.csv"
    if not xsec_path.exists():
        raise FileNotFoundError(f"Falta {xsec_path}")

    xsec = pd.read_csv(xsec_path)
    xsec.columns = [c.strip() for c in xsec.columns]

    col_q2lo = find_col(xsec, ["Q2low", "Q2Low", "Q2_low"])
    col_q2hi = find_col(xsec, ["Q2High", "Q2high", "Q2_hi", "Q2_high"])
    col_xsec = find_col(xsec, ["xsec", "XSec", "dsigma", "dsigdq2", "dsigma_dQ2"])

    q2_low = xsec[col_q2lo].to_numpy(float)
    q2_high = xsec[col_q2hi].to_numpy(float)
    data = xsec[col_xsec].to_numpy(float)
    n_bins = len(data)

    # ---- COV ----
    cov_path = raw_dir / "cov_tot.csv"
    if not cov_path.exists():
        raise FileNotFoundError(f"Falta {cov_path}")

    V = read_cov_matrix(cov_path, n_bins)

    # Typical MINERvA release: xsec numbers in 1e-38, cov in 1e-80 -> scale 1e-4
    COV_SCALE = 1e-4
    V = V * COV_SCALE

    # ---- FLUX ----
    flux_path = pick_flux_csv(raw_dir)
    flux = pd.read_csv(flux_path)
    flux.columns = [c.strip() for c in flux.columns]
    col_E = find_col(flux, ["Energy(GeV)", "Energy", "E", "enu"])
    col_phi = pick_flux_col(flux)

    flux_E = flux[col_E].to_numpy(float)
    flux_phi = flux[col_phi].to_numpy(float)

    print("Usando flujo:", flux_path.name)
    print("Columna E:", col_E)
    print("Columna phi:", col_phi)

    # ---- params for the model ----
    params = {
        "MA": 1.00,  # puedes probar 1.03, 1.35, etc.
        "MV2": 0.71, # dipole vector mass squared
    }

    # ---- MODEL: flux-folded + cuts ----
    model = flux_folded_binned_xsec(
        q2_low=q2_low,
        q2_high=q2_high,
        flux_E=flux_E,
        flux_phi=flux_phi,
        dsigma_dQ2_callable=dsigma_dQ2_model,
        params=params,
        nQ2=80,
        Ev_max=20.0,
    )

    # ---- CHI2 (correlated) ----
    r = data - model
    chi2 = r @ np.linalg.solve(V, r)
    ndof = n_bins

    print("chi2 =", float(chi2))
    print("chi2/ndof =", float(chi2 / ndof))

    # ---- SAVE TABLE ----
    out = xsec.copy()
    out["model"] = model
    out["residual"] = r
    out_path = proc_dir / "comparison_bins.csv"
    out.to_csv(out_path, index=False)
    print("Guardado:", out_path)

    # ---- PLOT ----
    q2_cent = 0.5 * (q2_low + q2_high)

    yerr = None
    try:
        col_stat = find_col(xsec, ["stat", "Stat", "staterr", "StatErr"])
        col_syst = find_col(xsec, ["syst", "Syst", "syserr", "SystErr"])
        yerr = np.sqrt(xsec[col_stat].to_numpy(float) ** 2 + xsec[col_syst].to_numpy(float) ** 2)
    except Exception:
        pass

    plt.figure()
    if yerr is None:
        plt.plot(q2_cent, data, marker="o", linestyle="none", label="MINERvA data")
    else:
        plt.errorbar(q2_cent, data, yerr=yerr, marker="o", linestyle="none", label="MINERvA data")

    plt.plot(q2_cent, model, label="Model (LS + dipole FF) (flux-folded + cuts)")
    plt.xlabel(r"$Q^2\ \mathrm{[GeV^2]}$")
    plt.ylabel(r"$\langle d\sigma/dQ^2\rangle\ [10^{-38}\ \mathrm{cm^2/GeV^2}]$")
    plt.legend()
    plt.tight_layout()

    fig1 = fig_dir / "minerva_hydrogen_dsig_dQ2_comparison.pdf"
    plt.savefig(fig1)
    print("Figura:", fig1)

    plt.figure()
    plt.axhline(1.0, linewidth=1)
    denom = np.where(np.abs(model) > 0, model, np.nan)
    plt.plot(q2_cent, data / denom, marker="o", linestyle="none")
    plt.xlabel(r"$Q^2\ \mathrm{[GeV^2]}$")
    plt.ylabel(r"Data / Model")
    plt.tight_layout()

    fig2 = fig_dir / "minerva_hydrogen_ratio_data_over_model.pdf"
    plt.savefig(fig2)
    print("Figura:", fig2)


if __name__ == "__main__":
    main()