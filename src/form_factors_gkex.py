# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 14:55:47 2026

@author: User
"""

# src/form_factors_gkex.py
# GKex(05) (Lomon): parameters from 2006 refit table, formulas from 2002 paper.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class GKex05Params:
    # Common constants (Table I)
    kappa_v: float = 3.706
    kappa_s: float = -0.12

    m_rho: float = 0.776
    m_omega: float = 0.784
    m_phi: float = 1.019
    m_rhop: float = 1.45
    m_omegap: float = 1.419

    # Fit parameters for GKex(05) (Table I)
    g_rhop_over_f_rhop: float = 0.007208
    kappa_rhop: float = 12.0

    g_omega_over_f_omega: float = 0.7021
    kappa_omega: float = 0.4027

    g_phi_over_f_phi: float = -0.1711
    kappa_phi: float = 0.01
    mu_phi: float = 0.2

    g_omegap_over_f_omegap: float = 0.164
    kappa_omegap: float = -2.973

    Lambda1: float = 0.93088
    LambdaD: float = 1.181
    Lambda2: float = 2.6115
    LambdaQCD: float = 0.150
    N: float = 1.0


def _qtilde2(Q2: float, LambdaD: float, LambdaQCD: float) -> float:
    """
    \tilde{Q}^2 = Q^2 * ln[(LambdaD^2 + Q^2)/LambdaQCD^2] / ln[LambdaD^2/LambdaQCD^2]
    Eq. (7) bottom line in Lomon 2002 paper.  (We use positive Q2 = |Q^2|.)
    """
    # protect logs at Q2=0
    if Q2 <= 0.0:
        return 0.0
    num = np.log((LambdaD**2 + Q2) / (LambdaQCD**2))
    den = np.log((LambdaD**2) / (LambdaQCD**2))
    return Q2 * (num / den)


def _F1_alpha(Q2: float, LambdaX: float, Lambda2: float, LambdaD: float, LambdaQCD: float) -> float:
    """
    F1^{alpha,D}(Q^2) = [Lambda_{1,D}^2/(Lambda_{1,D}^2 + \tilde{Q}^2)] * [Lambda2^2/(Lambda2^2 + \tilde{Q}^2)]
    Eq. (7) (first line) in Lomon 2002.
    """
    Qt2 = _qtilde2(Q2, LambdaD, LambdaQCD)
    return (LambdaX**2 / (LambdaX**2 + Qt2)) * (Lambda2**2 / (Lambda2**2 + Qt2))


def _F2_alpha(Q2: float, LambdaX: float, Lambda2: float, LambdaD: float, LambdaQCD: float) -> float:
    """
    F2^{alpha,D}(Q^2) = [Lambda_{1,D}^2/(Lambda_{1,D}^2 + \tilde{Q}^2)] * [Lambda2^2/(Lambda2^2 + \tilde{Q}^2)]^2
    Eq. (7) (second line) in Lomon 2002.
    """
    Qt2 = _qtilde2(Q2, LambdaD, LambdaQCD)
    return (LambdaX**2 / (LambdaX**2 + Qt2)) * (Lambda2**2 / (Lambda2**2 + Qt2))**2


def _F1_phi(Q2: float, Lambda1: float, Lambda2: float, LambdaD: float, LambdaQCD: float) -> float:
    """
    F1^phi(Q^2) = F1^alpha(Q^2) * (Q^2/(Lambda1^2 + Q^2))^{1.5}, with F1^phi(0)=0
    Eq. (7) (third line) in Lomon 2002.
    """
    if Q2 <= 0.0:
        return 0.0
    base = _F1_alpha(Q2, Lambda1, Lambda2, LambdaD, LambdaQCD)
    return base * (Q2 / (Lambda1**2 + Q2))**1.5


def _F2_phi(Q2: float, Lambda1: float, Lambda2: float, LambdaD: float, LambdaQCD: float, mu_phi: float) -> float:
    """
    F2^phi(Q^2) = F2^alpha(Q^2) * (mu_phi^2 Q^2 + mu_phi^2 Lambda1^2)/(mu_phi^2 Lambda1^2 + Q^2))^{1.5}
    Eq. (7) (fourth line) in Lomon 2002.
    """
    base = _F2_alpha(Q2, Lambda1, Lambda2, LambdaD, LambdaQCD)
    num = (mu_phi**2) * Q2 + (mu_phi**2) * (Lambda1**2)
    den = (mu_phi**2) * (Lambda1**2) + Q2
    return base * (num / den)**1.5


def F_is_iv(Q2: float, p: GKex05Params = GKex05Params()) -> tuple[float, float, float, float]:
    """
    Returns (F1^is, F2^is, F1^iv, F2^iv) for positive Q2=|Q^2|.
    Formulas are Eq. (6) in Lomon 2002 paper.
    Params are GKex(05) from Lomon 2006 Table I.
    """
    # Hadronic form factors for meson terms (use Lambda1)
    F1_rho = _F1_alpha(Q2, p.Lambda1, p.Lambda2, p.LambdaD, p.LambdaQCD)
    F2_rho = _F2_alpha(Q2, p.Lambda1, p.Lambda2, p.LambdaD, p.LambdaQCD)

    F1_omega = _F1_alpha(Q2, p.Lambda1, p.Lambda2, p.LambdaD, p.LambdaQCD)
    F2_omega = _F2_alpha(Q2, p.Lambda1, p.Lambda2, p.LambdaD, p.LambdaQCD)

    # For rho' and omega' use same hadronic FF shape but different pole masses in prefactor
    F1_rhop = F1_rho
    F2_rhop = F2_rho
    F1_omegap = F1_omega
    F2_omegap = F2_omega

    # Phi special
    F1_phi = _F1_phi(Q2, p.Lambda1, p.Lambda2, p.LambdaD, p.LambdaQCD)
    F2_phi = _F2_phi(Q2, p.Lambda1, p.Lambda2, p.LambdaD, p.LambdaQCD, p.mu_phi)

    # Quark-nucleon (pQCD) terms: use LambdaD as Lambda_{1,D}
    F1_D = _F1_alpha(Q2, p.LambdaD, p.Lambda2, p.LambdaD, p.LambdaQCD)
    F2_D = _F2_alpha(Q2, p.LambdaD, p.Lambda2, p.LambdaD, p.LambdaQCD)

    # --- isovector: Eq (6) first two lines for F1^iv, F2^iv ---
    A1 = (1.0317 + 0.0875 * (1.0 + Q2 / 0.3176)**(-2.0)) / (1.0 + Q2 / 0.5496)
    A2 = (5.7824 + 0.3907 * (1.0 + Q2 / 0.1422)**(-1.0)) / (1.0 + Q2 / 0.5362)

    F1_iv = (p.N / 2.0) * A1 * F1_rho \
            + (p.g_rhop_over_f_rhop) * (p.m_rhop**2 / (p.m_rhop**2 + Q2)) * F1_rhop \
            + (1.0 - 1.1192 * p.N / 2.0 - p.g_rhop_over_f_rhop) * F1_D

    F2_iv = (p.N / 2.0) * A2 * F2_rho \
            + (p.kappa_rhop * p.g_rhop_over_f_rhop) * (p.m_rhop**2 / (p.m_rhop**2 + Q2)) * F2_rhop \
            + (p.kappa_v - 6.1731 * p.N / 2.0 - p.kappa_rhop * p.g_rhop_over_f_rhop) * F2_D

    # --- isoscalar: Eq (6) last two lines for F1^is, F2^is ---
    F1_is = (p.g_omega_over_f_omega) * (p.m_omega**2 / (p.m_omega**2 + Q2)) * F1_omega \
            + (p.g_omegap_over_f_omegap) * (p.m_omegap**2 / (p.m_omegap**2 + Q2)) * F1_omegap \
            + (p.g_phi_over_f_phi) * (p.m_phi**2 / (p.m_phi**2 + Q2)) * F1_phi \
            + (1.0 - p.g_omega_over_f_omega - p.g_omegap_over_f_omegap) * F1_D

    F2_is = (p.kappa_omega * p.g_omega_over_f_omega) * (p.m_omega**2 / (p.m_omega**2 + Q2)) * F2_omega \
            + (p.kappa_omegap * p.g_omegap_over_f_omegap) * (p.m_omegap**2 / (p.m_omegap**2 + Q2)) * F2_omegap \
            + (p.kappa_phi * p.g_phi_over_f_phi) * (p.m_phi**2 / (p.m_phi**2 + Q2)) * F2_phi \
            + (p.kappa_s - p.kappa_omega * p.g_omega_over_f_omega - p.kappa_omegap * p.g_omegap_over_f_omegap - p.kappa_phi * p.g_phi_over_f_phi) * F2_D

    return F1_is, F2_is, F1_iv, F2_iv


def F1F2_pn(Q2: float, p: GKex05Params = GKex05Params()) -> tuple[float, float, float, float]:
    """
    Returns (F1p, F2p, F1n, F2n) using Eq. (5) from Lomon 2002:
    2F_i^p = F_i^is + F_i^iv,  2F_i^n = F_i^is - F_i^iv.
    """
    F1_is, F2_is, F1_iv, F2_iv = F_is_iv(Q2, p=p)
    F1p = 0.5 * (F1_is + F1_iv)
    F2p = 0.5 * (F2_is + F2_iv)
    F1n = 0.5 * (F1_is - F1_iv)
    F2n = 0.5 * (F2_is - F2_iv)
    return F1p, F2p, F1n, F2n


def sachs_gkex(Q2: float, M: float = 0.939565, p: GKex05Params = GKex05Params()) -> tuple[float, float, float, float]:
    """
    Returns (GEp, GMp, GEn, GMn) for positive Q2=|Q^2|.
    Sachs relations: GE = F1 - tau F2, GM = F1 + F2, tau=Q2/(4M^2) (Eq. (4) Lomon 2002).
    """
    F1p, F2p, F1n, F2n = F1F2_pn(Q2, p=p)
    tau = Q2 / (4 * M**2)
    GEp = F1p - tau * F2p
    GMp = F1p + F2p
    GEn = F1n - tau * F2n
    GMn = F1n + F2n
    return GEp, GMp, GEn, GMn