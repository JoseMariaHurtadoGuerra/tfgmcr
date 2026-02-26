# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 11:53:10 2026

@author: User
"""

import numpy as np

def wi_from_formfactors(Q2_abs, M, F1V, F2V, GA, FP):
    """Devuelve w1..w5 evaluados en |Q^2|=Q2_abs."""
    tau = Q2_abs / (4*M**2)

    w1 = GA**2 + tau * ((F1V + F2V)**2 + GA**2)
    w2 = (F1V**2) + GA**2 + tau * (F2V**2)
    w3 = 2.0 * GA * (F1V + F2V)

    w4 = -(GA * FP)/M + tau * (FP**2) + ((Q2_abs - 4*M**2)/((4*M**2)**2))*(F2V**2) + 1j*(F1V*F2V)/(2*M**2)

    w5 = w2
    return w1, w2, w3, w4, w5


def contraction_lep_had(Q2_abs, Ev, El, cos_th, M, ml, F1V, F2V, GA, FP, is_antinu=False):
    """
    Implementa exactamente tu captura para  η~_{μν} H~^{μν}.
    OJO: para antineutrino, el término con w3 cambia de signo.
    """
    kl = np.sqrt(max(El**2 - ml**2, 0.0))

    w1, w2, w3, w4, w5 = wi_from_formfactors(Q2_abs, M, F1V, F2V, GA, FP)

    # cambio de signo ν vs νbar en el término antisymétrico (w3)
    s = -1.0 if is_antinu else +1.0

    A = (El - kl*cos_th)
    B = (El + kl*cos_th)

    X = 2.0*w1*Ev*A \
        +      w2*Ev*B \
        + s*(w3/M)*Ev*((Ev+El)*A - ml**2) \
        +      w4*(ml**2)*Ev*A \
        - (w5/M)*Ev*(ml**2)

    return X