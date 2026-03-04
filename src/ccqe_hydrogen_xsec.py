# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:10:22 2026

@author: User
"""

# src/ccqe_hydrogen_xsec.py
from __future__ import annotations
import numpy as np

# ---------- Constants (GeV-based natural units) ----------
GF = 1.1663787e-5          # GeV^-2
COS_TC = 0.9740            # cos(theta_C)

MU_P = 2.793
MU_N = -1.913

MP = 0.9382720813
MN = 0.9395654133
M = 0.5 * (MP + MN)
M_MU = 0.1056583745

GEV2_TO_CM2 = 0.3893793721e-27  # 1 GeV^-2 -> cm^2


# -----------------------
# Vector form factors
# -----------------------
def _dipole_GD(Q2: float, MV2: float) -> float:
    return 1.0 / (1.0 + Q2 / MV2) ** 2


def _vector_sachs_dipole(Q2: float, MV2: float):
    GD = _dipole_GD(Q2, MV2)
    GEp = GD
    GEn = 0.0
    GMp = MU_P * GD
    GMn = MU_N * GD
    return float(GEp), float(GMp), float(GEn), float(GMn)


_GKEX_LOADED = False
_GKEX = None
_GKEX_PACK = None  # function that returns sachs


def _parse_sachs_output(vals):
    """
    Normaliza lo que devuelva sachs_gkex(Q2) a (GEp,GMp,GEn,GMn).

    Acepta:
      - tupla/lista de longitud 4: (GEp,GMp,GEn,GMn)
      - dict con claves tipo 'GEp','GMp','GEn','GMn' (varias variantes)
      - tupla de 2 tuplas: ((GEp,GEn),(GMp,GMn)) o ((GEp,GMp),(GEn,GMn)) -> lo intentamos inferir
    """
    # dict
    if isinstance(vals, dict):
        keys = {k.lower(): k for k in vals.keys()}

        def get_any(*cands):
            for c in cands:
                cl = c.lower()
                if cl in keys:
                    return float(vals[keys[cl]])
            return None

        GEp = get_any("GEp", "Gep", "G_Ep", "G_E_p")
        GMp = get_any("GMp", "Gmp", "G_Mp", "G_M_p")
        GEn = get_any("GEn", "Gen", "G_En", "G_E_n")
        GMn = get_any("GMn", "Gmn", "G_Mn", "G_M_n")

        if None not in (GEp, GMp, GEn, GMn):
            return GEp, GMp, GEn, GMn

        raise ValueError(f"sachs_gkex devolvió dict pero no encuentro claves Sachs. Claves: {list(vals.keys())}")

    # list/tuple
    if isinstance(vals, (list, tuple)):
        if len(vals) == 4:
            return tuple(float(x) for x in vals)

        # casos con anidamiento en 2 bloques
        if len(vals) == 2 and all(isinstance(v, (list, tuple)) for v in vals):
            a, b = vals
            if len(a) == 2 and len(b) == 2:
                a0, a1 = float(a[0]), float(a[1])
                b0, b1 = float(b[0]), float(b[1])

                # Intento 1: ((GEp,GEn),(GMp,GMn))
                # suele dar GE de orden ~1 y GM de orden ~2-3 (a bajo Q2)
                # así que si |b| tiende a ser mayor, b puede ser GM.
                if abs(b0) + abs(b1) > abs(a0) + abs(a1):
                    GEp, GEn = a0, a1
                    GMp, GMn = b0, b1
                    return GEp, GMp, GEn, GMn

                # Intento 2: ((GEp,GMp),(GEn,GMn))
                # asumimos orden por p/n
                GEp, GMp = a0, a1
                GEn, GMn = b0, b1
                return GEp, GMp, GEn, GMn

        raise ValueError(f"sachs_gkex devolvió una estructura no soportada: {vals}")

    raise ValueError("Salida de sachs_gkex no soportada.")


def _load_gkex():
    global _GKEX_LOADED, _GKEX, _GKEX_PACK
    if _GKEX_LOADED:
        return
    _GKEX_LOADED = True

    try:
        import form_factors_gkex as gkex
    except Exception:
        _GKEX = None
        _GKEX_PACK = None
        return

    _GKEX = gkex

    # En tu módulo, según el error, existe sachs_gkex:
    if hasattr(gkex, "sachs_gkex") and callable(getattr(gkex, "sachs_gkex")):
        _GKEX_PACK = getattr(gkex, "sachs_gkex")
        return

    # fallback por si el nombre cambiara
    for name in ["sachs", "ff_sachs", "gkex_sachs", "GKeX_sachs"]:
        if hasattr(gkex, name) and callable(getattr(gkex, name)):
            _GKEX_PACK = getattr(gkex, name)
            return

    callables = [n for n in dir(gkex) if callable(getattr(gkex, n)) and not n.startswith("_")]
    raise ImportError(
        "He encontrado src/form_factors_gkex.py pero no localizo una función Sachs.\n"
        f"Callables disponibles: {callables}"
    )


def _vector_sachs_gkex(Q2: float):
    _load_gkex()
    if _GKEX_PACK is None:
        raise ImportError("No se ha podido cargar sachs_gkex desde src/form_factors_gkex.py.")
    vals = _GKEX_PACK(float(Q2))
    return _parse_sachs_output(vals)


def _vector_sachs(Q2: float, MV2: float, vector_ff: str):
    vf = vector_ff.lower().strip()
    if vf == "dipole":
        return _vector_sachs_dipole(Q2, MV2)
    if vf == "gkex":
        return _vector_sachs_gkex(Q2)
    raise ValueError("vector_ff debe ser 'dipole' o 'gkex'.")


def _F1V_xiF2V_from_sachs(Q2: float, MV2: float, vector_ff: str):
    """
    Isovector Sachs:
      GVE = GEp - GEn
      GVM = GMp - GMn

    Convert:
      F1V = (GVE + tau*GVM)/(1+tau)
      xiF2V = (GVM - GVE)/(1+tau)
    """
    GEp, GMp, GEn, GMn = _vector_sachs(Q2, MV2, vector_ff)
    GVE = GEp - GEn
    GVM = GMp - GMn

    tau = Q2 / (4.0 * M * M)
    denom = 1.0 + tau
    F1V = (GVE + tau * GVM) / denom
    xiF2V = (GVM - GVE) / denom
    return F1V, xiF2V, tau


# -----------------------
# Axial form factor (dipole)
# -----------------------
def _FA_dipole(Q2: float, MA: float, gA: float = -1.267) -> float:
    return gA / (1.0 + Q2 / (MA * MA)) ** 2


# -----------------------
# CCQE: dσ/dQ² for \barνμ p → μ+ n
# -----------------------
def dsigma_dQ2_numubar_p(
    Ev: float,
    Q2: float,
    MA: float = 1.00,
    MV2: float = 0.71,
    vector_ff: str = "gkex",
) -> float:
    """
    dσ/dQ² for  \bar{ν}_μ + p → μ^+ + n (free proton at rest),
    Llewellyn–Smith A,B,C form.

    Returns: [1e-38 cm² / GeV²]
    """
    if Ev <= 0.0 or Q2 <= 0.0:
        return 0.0

    ml = M_MU
    if Q2 >= 4.0 * M * Ev:
        return 0.0

    F1V, xiF2V, tau = _F1V_xiF2V_from_sachs(Q2, MV2, vector_ff)
    FA = _FA_dipole(Q2, MA)

    prefA = (ml * ml + Q2) / (4.0 * M * M)

    A = prefA * (
        (4.0 + Q2 / (M * M)) * (FA * FA)
        - (4.0 - Q2 / (M * M)) * (F1V * F1V)
        + (Q2 / (M * M)) * (xiF2V * xiF2V) * (1.0 - tau)
        + 4.0 * (Q2 / (M * M)) * (F1V * xiF2V)
    )

    B = (Q2 / (M * M)) * (FA * (F1V + xiF2V))
    C = 0.25 * ((FA * FA) + (F1V * F1V) + tau * (xiF2V * xiF2V))

    su = 4.0 * M * Ev - Q2 - ml * ml
    pref = (M * M) * (GF * GF) * (COS_TC * COS_TC) / (8.0 * np.pi * Ev * Ev)

    # antineutrino
    term = A + (su * B) / (M * M) + C * (su * su) / (M ** 4)

    dsig_dQ2_GeV4 = pref * term
    dsig_dQ2_cm2_GeV2 = dsig_dQ2_GeV4 * GEV2_TO_CM2
    return dsig_dQ2_cm2_GeV2 / 1e-38