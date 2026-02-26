# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccqe_contraction import contraction_lep_had  # usa tu función ya implementada en src

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ---------- Constantes (GeV, unidades naturales) ----------
GF = 1.1663787e-5
cosC = 0.97420
M = 0.939565
m_mu = 0.105658

GEV2_TO_CM2 = 0.389379e-27  # 1 GeV^-2 -> cm^2


# ---------- Form factors (Galster + axial dipole + pion pole) ----------
def G_D(Q2_abs, MV=0.843):
    return 1.0 / (1.0 + Q2_abs / MV**2)**2

def sachs_galster(Q2_abs):
    mu_p = 2.79284734463
    mu_n = -1.91304273
    tau = Q2_abs / (4*M**2)

    GD = G_D(Q2_abs)
    GEp = GD
    GMp = mu_p * GD
    GMn = mu_n * GD

    lam = 5.6
    xi_n = 1.0 / (1.0 + lam*tau)
    GEn = -mu_n * tau * GD * xi_n
    return GEp, GMp, GEn, GMn

def F1F2_from_GE_GM(GE, GM, Q2_abs):
    tau = Q2_abs / (4*M**2)
    F1 = (GE + tau*GM) / (1.0 + tau)
    F2 = (GM - GE) / (1.0 + tau)
    return F1, F2

def isovector_F1F2_from_sachs(GEp, GMp, GEn, GMn, Q2_abs):
    F1p, F2p = F1F2_from_GE_GM(GEp, GMp, Q2_abs)
    F1n, F2n = F1F2_from_GE_GM(GEn, GMn, Q2_abs)
    return (F1p - F1n), (F2p - F2n)

def GA_dipole(Q2_abs, MA=1.03, gA=1.267):
    return gA / (1.0 + Q2_abs/MA**2)**2

def FP_pionpole(Q2_abs, GA, mpi=0.13957):
    return (2*M**2 * GA) / (mpi**2 + Q2_abs)

def vector_form_factors(Q2_abs, model="galster"):
    if model == "galster":
        GEp, GMp, GEn, GMn = sachs_galster(Q2_abs)
        return isovector_F1F2_from_sachs(GEp, GMp, GEn, GMn, Q2_abs)
    
    elif model == "gkex":
     from form_factors_gkex import sachs_gkex
    GEp, GMp, GEn, GMn = sachs_gkex(Q2_abs, M=M)
    return isovector_F1F2_from_sachs(GEp, GMp, GEn, GMn, Q2_abs)


# ---------- Cinemática QE ----------
def q2_abs(Ev, El, cos_th):
    kl = np.sqrt(max(El**2 - m_mu**2, 0.0))
    return 2*Ev*(El - kl*cos_th) - m_mu**2  # |Q^2|

def qe_equation(El, Ev, cos_th):
    return 2*M*(Ev - El) - q2_abs(Ev, El, cos_th)

def solve_El(Ev, cos_th):
    El_min = m_mu
    El_max = Ev + M

    fmin = qe_equation(El_min, Ev, cos_th)
    fmax = qe_equation(El_max, Ev, cos_th)
    if fmin * fmax > 0:
        return None

    return brentq(qe_equation, El_min, El_max, args=(Ev, cos_th), maxiter=200)

def f_rec(Ev, El, cos_th):
    kl = np.sqrt(El**2 - m_mu**2)
    return 1.0 + Ev * (kl - El*cos_th) / (M * kl)


# ---------- Sección eficaz (2.84) ----------
def dsigma_dOmega(Ev, cos_th, vector_model="galster", MA=1.03, is_antinu=False):
    El = solve_El(Ev, cos_th)
    if El is None:
        return 0.0

    kl = np.sqrt(El**2 - m_mu**2)
    Q2 = q2_abs(Ev, El, cos_th)
    frec = f_rec(Ev, El, cos_th)

    F1V, F2V = vector_form_factors(Q2, model=vector_model)
    GA = GA_dipole(Q2, MA=MA)
    FP = FP_pionpole(Q2, GA)

    # IMPORTANTE: esto debe coincidir con la firma de tu función en src
    X = contraction_lep_had(Q2, Ev, El, cos_th, M, m_mu, F1V, F2V, GA, FP, is_antinu=is_antinu)

    pref = (GF**2 * cosC**2) / (4*np.pi**2)
    return pref * (kl/Ev) * (1.0/frec) * X

def main():
    Ev = 1.0  # GeV
    thetas = np.linspace(0.0, np.pi, 181)
    coss = np.cos(thetas)

    # --- curvas: Galster y (si existe) GKeX ---
    y_red = np.array([
        dsigma_dOmega(Ev, c, vector_model="galster", MA=1.03, is_antinu=False)
        for c in coss
    ]) * GEV2_TO_CM2

    # Intentamos GKeX: si aún no lo has implementado, no rompe el script
    y_blue = None
    y_green = None
    try:
        y_blue = np.array([
            dsigma_dOmega(Ev, c, vector_model="gkex", MA=1.03, is_antinu=False)
            for c in coss
        ]) * GEV2_TO_CM2

        y_green = np.array([
            dsigma_dOmega(Ev, c, vector_model="gkex", MA=1.35, is_antinu=False)
            for c in coss
        ]) * GEV2_TO_CM2

    except NotImplementedError:
        print("GKeX aún no implementado: se plotea solo Galster.")

    # --- plot ---
    plt.figure()
    plt.plot(thetas * 180 / np.pi, y_red, label="Galster (MA=1.03)")

    if y_blue is not None:
        plt.plot(thetas * 180 / np.pi, y_blue, "--", label="GKeX (MA=1.03)")

    if y_green is not None:
        plt.plot(thetas * 180 / np.pi, y_green, label="GKeX (MA=1.35)")

    plt.xlabel(r'$\theta_\mu$ [deg]')
    plt.ylabel(r'$d\sigma/d\Omega$ [cm$^2$/sr]')
    plt.legend()
    plt.tight_layout()

    # --- guardado robusto (ruta absoluta) ---
    from pathlib import Path
    outdir = Path(r"C:\Users\User\Documents\tfgmcr\results\figures")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "fig_4_1.pdf"
    print("saving to:", outfile)
    plt.savefig(outfile)

    plt.show()



if __name__ == "__main__":
    main()