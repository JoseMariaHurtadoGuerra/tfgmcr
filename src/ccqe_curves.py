# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:02:52 2026

@author: User
"""

# src/ccqe_curves.py
import numpy as np

# IMPORTA desde donde las tengas ahora:
# - dsigma_dOmega(Ev, cos_th, vector_model, MA, is_antinu)
# - solve_El(Ev, cos_th)
# - q2_abs(Ev, El, cos_th)
# - GEV2_TO_CM2
from scripts.make_fig4_1 import dsigma_dOmega, solve_El, q2_abs, GEV2_TO_CM2


def curve_theta(Ev, vector_model="gkex", MA=1.03, is_antinu=False, npts=361):
    thetas = np.linspace(0.0, np.pi, npts)
    coss = np.cos(thetas)
    y = np.array([
        np.real(dsigma_dOmega(Ev, c, vector_model=vector_model, MA=MA, is_antinu=is_antinu))
        for c in coss
    ]) * GEV2_TO_CM2
    return thetas * 180/np.pi, y


def curve_q2_reparam(Ev, vector_model="gkex", MA=1.03, is_antinu=False, npts=721):
    thetas = np.linspace(0.0, np.pi, npts)
    coss = np.cos(thetas)

    Q2_list, y_list = [], []
    for c in coss:
        El = solve_El(Ev, c)
        if El is None:
            continue
        Q2 = q2_abs(Ev, El, c)
        y = np.real(dsigma_dOmega(Ev, c, vector_model=vector_model, MA=MA, is_antinu=is_antinu))
        if np.isfinite(Q2) and np.isfinite(y) and Q2 > 0:
            Q2_list.append(Q2)
            y_list.append(y)

    Q2 = np.array(Q2_list)
    y = np.array(y_list) * GEV2_TO_CM2
    idx = np.argsort(Q2)
    return Q2[idx], y[idx]