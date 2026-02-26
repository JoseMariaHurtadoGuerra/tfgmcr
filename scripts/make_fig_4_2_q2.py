# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 15:29:19 2026

@author: User
"""

# scripts/make_fig4_2.py
# Figura 4.2 estilo Guillermo: dσ/dΩ vs |Q^2| (izq) y vs θ (der), para varias energías

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- paths ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))  # para importar make_fig4_1.py

# Importamos lo ya implementado y funcionando en tu make_fig4_1.py
# Ajusta el nombre si tu archivo se llama distinto
from make_fig4_1 import dsigma_dOmega, solve_El, q2_abs, GEV2_TO_CM2


def dsdo_vs_theta(Ev, vector_model, MA, is_antinu=False, npts=721):
    """Devuelve (theta_deg, dsdo_cm2sr) para un Ev fijo."""
    thetas = np.linspace(0.0, np.pi, npts)
    coss = np.cos(thetas)

    dsdo = np.array([
        np.real(dsigma_dOmega(Ev, c, vector_model=vector_model, MA=MA, is_antinu=is_antinu))
        for c in coss
    ]) * GEV2_TO_CM2

    return thetas * 180.0 / np.pi, dsdo


def dsdo_vs_q2(Ev, vector_model, MA, is_antinu=False, npts=721):
    """
    Devuelve (Q2_sorted, dsdo_sorted) donde dsdo = dσ/dΩ.
    Es la MISMA curva dσ/dΩ(θ) pero representada contra Q2(θ).
    """
    thetas = np.linspace(0.0, np.pi, npts)
    coss = np.cos(thetas)

    Q2_list = []
    dsdo_list = []

    for c in coss:
        El = solve_El(Ev, c)
        if El is None:
            continue

        Q2 = q2_abs(Ev, El, c)  # |Q^2|
        dsdo = np.real(dsigma_dOmega(Ev, c, vector_model=vector_model, MA=MA, is_antinu=is_antinu))

        if np.isfinite(Q2) and np.isfinite(dsdo) and Q2 > 0:
            Q2_list.append(Q2)
            dsdo_list.append(dsdo)

    Q2v = np.array(Q2_list)
    dsdo = np.array(dsdo_list) * GEV2_TO_CM2

    idx = np.argsort(Q2v)
    return Q2v[idx], dsdo[idx]


def main():
    outdir = ROOT / "results" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    # Energías como en la figura de Guillermo
    Ev_list = [0.5, 1.0, 1.5]

    # Curvas: (label, vector_model, MA, linestyle)
    curves = [
        ("Galster", "galster", 1.03, "-"),
        ("GKeX",    "gkex",    1.03, "--"),
        (r"GKeX ($M_A=1.35$ GeV)", "gkex", 1.35, "-"),
    ]

    fig, axes = plt.subplots(
        nrows=len(Ev_list), ncols=2,
        figsize=(10, 9),
        sharey="row"
    )

    for i, Ev in enumerate(Ev_list):
        ax_q2 = axes[i, 0]
        ax_th = axes[i, 1]

        # --- columna izquierda: dσ/dΩ vs |Q^2| ---
        for label, vmodel, MA, ls in curves:
            Q2, dsdo = dsdo_vs_q2(Ev, vmodel, MA, is_antinu=False, npts=721)
            ax_q2.plot(Q2, dsdo, ls, label=label)

        ax_q2.set_title(rf"$E_\nu = {Ev:.1f}\ \mathrm{{GeV}}$")
        ax_q2.set_xlabel(r"$|Q^2|\ (\mathrm{GeV}^2)$")
        ax_q2.grid(True, alpha=0.3)

        # --- columna derecha: dσ/dΩ vs θ ---
        for label, vmodel, MA, ls in curves:
            th_deg, dsdo = dsdo_vs_theta(Ev, vmodel, MA, is_antinu=False, npts=721)
            ax_th.plot(th_deg, dsdo, ls, label=label)

        ax_th.set_title(rf"$E_\nu = {Ev:.1f}\ \mathrm{{GeV}}$")
        ax_th.set_xlabel(r"$\theta_\mu\ (\mathrm{deg})$")
        ax_th.grid(True, alpha=0.3)

        # Etiqueta Y en cada fila (como en Guillermo, misma magnitud en ambas columnas)
        ax_q2.set_ylabel(r"$d\sigma/d\Omega\ (\mathrm{cm}^2/\mathrm{sr})$")

    # Leyenda en el panel inferior derecho (o donde prefieras)
    axes[-1, -1].legend(loc="best")

    fig.tight_layout()
    outfile = outdir / "fig_4_2_like_guillermo.pdf"
    fig.savefig(outfile)
    print("Saved:", outfile)
    plt.show()


if __name__ == "__main__":
    main()