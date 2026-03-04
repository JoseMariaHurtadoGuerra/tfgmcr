# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:02:35 2026

@author: User
"""

# src/minerva/flux_folding.py
from __future__ import annotations
import numpy as np

# --- masses (GeV) ---
MP = 0.9382720813
MN = 0.9395654133
M_MU = 0.1056583745
DEG = np.pi / 180.0


def omega_of_Q2(Q2: float, Mp: float = MP, Mn: float = MN) -> float:
    """Elastic CC: omega = (Q2 + Mn^2 - Mp^2)/(2 Mp)."""
    return (Q2 + (Mn * Mn - Mp * Mp)) / (2.0 * Mp)


def muon_kinematics(Ev: float, Q2: float, Mp: float = MP, Mn: float = MN, m_mu: float = M_MU):
    """
    From (Ev,Q2) with proton at rest (2->2 elastic CC), returns (E_mu, cos(theta_mu)).
    If unphysical: (None,None).
    """
    w = omega_of_Q2(Q2, Mp=Mp, Mn=Mn)
    E_mu = Ev - w
    if E_mu <= m_mu:
        return None, None

    p_mu = np.sqrt(max(E_mu * E_mu - m_mu * m_mu, 0.0))
    if p_mu <= 0:
        return None, None

    # Q2 = 2 Ev (E_mu - p_mu cosθ) - m_mu^2
    cos_th = (Ev * E_mu - 0.5 * (Q2 + m_mu * m_mu)) / (Ev * p_mu)
    if (not np.isfinite(cos_th)) or (cos_th < -1.0) or (cos_th > 1.0):
        return None, None

    return E_mu, cos_th


def passes_minos_cuts(E_mu: float | None, cos_th: float | None) -> bool:
    """MINOS-like cuts: 1.5<E_mu<20 GeV and theta<20deg."""
    if E_mu is None or cos_th is None:
        return False
    theta = np.arccos(np.clip(cos_th, -1.0, 1.0))
    return (1.5 < E_mu < 20.0) and (theta < 20.0 * DEG)


def flux_folded_binned_xsec(
    q2_low: np.ndarray,
    q2_high: np.ndarray,
    flux_E: np.ndarray,
    flux_phi: np.ndarray,
    dsigma_dQ2_callable,
    params: dict,
    nQ2: int = 80,
    Ev_max: float = 20.0,
) -> np.ndarray:
    """
    Flux-folded and cut-applied bin-averaged <dσ/dQ2>:

      pred_bin = (1/ΔQ2) * (1/Φ_tot) ∫ dE φ(E) ∫_{bin} dQ2 [dσ/dQ2(E,Q2)] * cuts

    NOTE: uses np.trapezoid (NumPy 2.x safe).
    """
    # integration helper (NumPy 2.x)
    trap = np.trapezoid

    flux_E = np.asarray(flux_E, dtype=float)
    flux_phi = np.asarray(flux_phi, dtype=float)

    m = (flux_E > 0.0) & (flux_E < Ev_max) & np.isfinite(flux_phi) & (flux_phi > 0.0)
    E = flux_E[m]
    phi = flux_phi[m]
    if len(E) < 5:
        raise ValueError("Flujo vacío/mal leído. Revisa el CSV del flujo y sus columnas.")

    phi_tot = trap(phi, E)
    if phi_tot <= 0:
        raise ValueError("Normalización de flujo <=0. Revisa el CSV del flujo.")

    preds = np.zeros(len(q2_low), dtype=float)

    for i, (lo, hi) in enumerate(zip(q2_low, q2_high)):
        lo = float(lo)
        hi = float(hi)
        q2_grid = np.linspace(lo, hi, nQ2)

        integrand_E = np.zeros_like(E)
        for j, (Ev, w_flux) in enumerate(zip(E, phi)):
            vals = np.zeros_like(q2_grid)

            for k, Q2 in enumerate(q2_grid):
                E_mu, cos_th = muon_kinematics(float(Ev), float(Q2))
                if not passes_minos_cuts(E_mu, cos_th):
                    vals[k] = 0.0
                else:
                    vals[k] = float(dsigma_dQ2_callable(float(Ev), float(Q2), params))

            int_Q2 = trap(vals, q2_grid)
            integrand_E[j] = w_flux * int_Q2

        num = trap(integrand_E, E)
        preds[i] = (num / phi_tot) / (hi - lo)

    return preds