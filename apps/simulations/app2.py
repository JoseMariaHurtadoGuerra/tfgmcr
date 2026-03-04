# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:27:13 2026

@author: User
"""

# apps/simulations/app2.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False
    import matplotlib.pyplot as plt


# -----------------------
# Paths
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../tfgmcr
SRC_DIR = PROJECT_ROOT / "src"
REFS_DIR = PROJECT_ROOT / "refs" / "minerva_hydrogen"
sys.path.insert(0, str(SRC_DIR))

from minerva.flux_folding import flux_folded_binned_xsec
from ccqe_hydrogen_xsec import dsigma_dQ2_numubar_p


# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Simulación 2 — MINERvA H", layout="wide")

# KaTeX: scroll horizontal si una ecuación es larga
st.markdown(
    """
<style>
.katex-display { overflow-x: auto; overflow-y: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------
# Helpers
# -----------------------
def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = list(df.columns)
    strip_map = {c.strip(): c for c in cols}

    for cand in candidates:
        if cand in strip_map:
            return strip_map[cand]

    lower_exact = {c.strip().lower(): c for c in cols}
    for cand in candidates:
        cl = cand.strip().lower()
        if cl in lower_exact:
            return lower_exact[cl]

    cols_low = [(c, c.strip().lower()) for c in cols]
    for cand in candidates:
        cl = cand.strip().lower()
        for orig, low in cols_low:
            if cl in low:
                return orig

    raise KeyError(f"No encuentro columnas {candidates}. Tengo: {list(df.columns)}")


def pick_flux_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.strip().lower().startswith("flux("):
            return c
    return find_col(df, ["flux", "phi"])


def read_cov_matrix(path: Path, n: int) -> np.ndarray:
    V = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if V.shape == (n, n + 1):
        V = V[:, 1:]
    if V.shape == (n + 1, n + 1):
        V = V[1:, 1:]
    if V.shape != (n, n):
        raise ValueError(f"Covarianza con forma {V.shape}, esperaba {(n, n)}. Revisa {path.name}.")
    return V


def dsigma_dQ2_model(Ev: float, Q2: float, params: dict) -> float:
    MA = float(params.get("MA", 1.00))
    MV2 = float(params.get("MV2", 0.71))
    vector_ff = str(params.get("vector_ff", "gkex"))
    return float(dsigma_dQ2_numubar_p(Ev, Q2, MA=MA, MV2=MV2, vector_ff=vector_ff))


@st.cache_data(show_spinner=False)
def load_inputs():
    need = [
        REFS_DIR / "hydrogen_xsec.csv",
        REFS_DIR / "cov_tot.csv",
        REFS_DIR / "flux_rhc_numubar_nueconstrained.csv",
    ]
    missing = [p for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos en refs/minerva_hydrogen:\n" + "\n".join([str(p) for p in missing])
        )

    xsec = pd.read_csv(REFS_DIR / "hydrogen_xsec.csv")
    xsec.columns = [c.strip() for c in xsec.columns]

    flux = pd.read_csv(REFS_DIR / "flux_rhc_numubar_nueconstrained.csv")
    flux.columns = [c.strip() for c in flux.columns]

    return xsec, flux


@st.cache_data(show_spinner=True)
def compute_fluxfolded_prediction(MA: float, MV2: float, vector_ff: str, nQ2: int, Ev_max: float):
    xsec, flux = load_inputs()

    col_q2lo = find_col(xsec, ["Q2low", "Q2Low", "Q2_low"])
    col_q2hi = find_col(xsec, ["Q2High", "Q2high", "Q2_hi", "Q2_high"])
    col_xsec = find_col(xsec, ["xsec", "XSec", "dsigma", "dsigdq2", "dsigma_dQ2"])

    q2_low = xsec[col_q2lo].to_numpy(float)
    q2_high = xsec[col_q2hi].to_numpy(float)
    data = xsec[col_xsec].to_numpy(float)
    q2_cent = 0.5 * (q2_low + q2_high)

    col_E = find_col(flux, ["Energy(GeV)", "Energy", "E", "enu"])
    col_phi = pick_flux_col(flux)
    flux_E = flux[col_E].to_numpy(float)
    flux_phi = flux[col_phi].to_numpy(float)

    params = {"MA": MA, "MV2": MV2, "vector_ff": vector_ff}

    model = flux_folded_binned_xsec(
        q2_low=q2_low,
        q2_high=q2_high,
        flux_E=flux_E,
        flux_phi=flux_phi,
        dsigma_dQ2_callable=dsigma_dQ2_model,
        params=params,
        nQ2=nQ2,
        Ev_max=Ev_max,
    )

    return q2_cent, q2_low, q2_high, data, model


def compute_chi2(data: np.ndarray, model: np.ndarray) -> tuple[float, float]:
    V = read_cov_matrix(REFS_DIR / "cov_tot.csv", len(data))
    V = V * 1e-4  # cov en 1e-80, xsec en 1e-38
    r = data - model
    chi2 = float(r @ np.linalg.solve(V, r))
    return chi2, chi2 / len(r)


# -----------------------
# UI
# -----------------------
st.title("Simulación 2 — CCQE en Hidrógeno y comparación con MINERvA")

st.markdown("Esta simulación reproduce la distribución **flux-integrated** de la sección eficaz diferencial medida por MINERvA:")
st.latex(r"\left\langle \frac{d\sigma}{dQ^2}\right\rangle")
st.markdown("para el proceso cuasi-elástico en hidrógeno:")
st.latex(r"\bar{\nu}_\mu + p \rightarrow \mu^+ + n")


tabs = st.tabs([
    "Introducción y objetivos",
    "Marco teórico",
    "Derivación de dσ/dQ²",
    "Promedio en flujo y cortes",
    "Comparación con datos",
    "Simulador"
])


# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Parámetros interactivos")

    MA = st.slider("M_A [GeV]", 0.80, 1.50, 1.00, 0.01)
    MV2 = st.slider("M_V² [GeV²] (dipolo)", 0.40, 1.20, 0.71, 0.01)

    vector_ff = st.selectbox(
        "FF vectoriales",
        options=["gkex", "dipole"],
        index=0,
        format_func=lambda x: "GKeX (CVC)" if x == "gkex" else "Dipolo",
    )

    st.divider()
    st.header("Cálculo numérico")
    nQ2 = st.slider("Puntos de integración Q² por bin", 20, 140, 80, 10)
    Ev_max = st.slider("Eν máx [GeV] (integración de flujo)", 5.0, 40.0, 20.0, 1.0)

    st.divider()
    st.header("Gráfica ratio")
    cap_ratio = st.checkbox("Limitar eje Y del ratio", value=True)
    ycap = st.slider("Ymax (Data/Model)", 1.5, 10.0, 3.0, 0.1)

    st.divider()
    st.header("Exportar")
    show_download = st.checkbox("Mostrar botón de descarga CSV", value=True)


# -----------------------
# Tab 0
# -----------------------
with tabs[0]:
    st.subheader("Motivación")
    st.markdown(
        "La dispersión CCQE en hidrógeno es un laboratorio ideal porque elimina gran parte de las incertidumbres nucleares. "
        "Esto permite estudiar con claridad la dependencia en \(Q^2\) de la corriente hadrónica, "
        "y en particular la sensibilidad al factor de forma axial."
    )

    st.subheader("Objetivos de la simulación")
    st.markdown(
        """
1. Construir un modelo reproducible para el observable experimental \(\langle d\sigma/dQ^2\rangle\) en hidrógeno.
2. Implementar el promedio en flujo del haz y los cortes cinemáticos del muón.
3. Comparar con datos MINERvA mediante un ajuste cuantitativo con covarianza total.
4. Estudiar sensibilidad a parámetros: \(M_A\) y el modelo de FF vectoriales (Dipolo vs GKeX).
"""
    )


# -----------------------
# Tab 1
# -----------------------
with tabs[1]:
    st.subheader("Definiciones")
    st.markdown("Definimos el cuatro-momento transferido:")
    st.latex(r"q^\mu = k^\mu - k'^\mu")
    st.markdown("y su invariante:")
    st.latex(r"Q^2 \equiv -q^2 > 0")

    st.subheader("Corriente hadrónica y factores de forma")
    st.markdown("La estructura del nucleón se codifica en factores de forma vectoriales y axiales.")
    st.latex(r"F_A(Q^2)=\frac{g_A}{\left(1+Q^2/M_A^2\right)^2}")
    st.markdown(
        "En esta app, los factores vectoriales se implementan mediante CVC en dos variantes: Dipolo o GKeX."
    )


# -----------------------
# Tab 2
# -----------------------
with tabs[2]:
    st.subheader("De la amplitud a la sección eficaz")
    st.markdown("La amplitud a nivel árbol tiene la estructura:")
    st.latex(
        r"\mathcal{M}\propto \bar{u}(k')\,\gamma^\mu(1-\gamma_5)\,u(k)\;\; \bar{u}(p')\,\Gamma_\mu(Q^2)\,u(p)"
    )
    st.markdown("El cuadrado de la amplitud lleva a la contracción:")
    st.latex(r"|\mathcal{M}|^2 \propto L_{\mu\nu}\,W^{\mu\nu}")

    st.subheader("Forma Llewellyn–Smith (nucleón libre)")
    st.markdown("Una forma estándar del resultado final para nucleón libre es:")
    st.latex(
        r"\frac{d\sigma}{dQ^2}=\frac{M^2G_F^2\cos^2\theta_C}{8\pi E_\nu^2}"
        r"\left[ A(Q^2)+\frac{(s-u)}{M^2}B(Q^2)+\frac{(s-u)^2}{M^4}C(Q^2)\right]"
    )
    st.markdown("donde \(A,B,C\) dependen de los FF vectoriales y axiales.")


# -----------------------
# Tab 3
# -----------------------
with tabs[3]:
    st.subheader("Por qué esta gráfica no es a energía fija")
    st.markdown("El experimento publica un observable integrado en flujo, no a un \(E_\nu\) fijo.")
    st.latex(
        r"\left\langle \frac{d\sigma}{dQ^2}\right\rangle"
        r"=\frac{1}{\Phi_{\rm tot}}\int dE_\nu\,\Phi(E_\nu)\,\frac{d\sigma}{dQ^2}(E_\nu,Q^2)\,\mathcal{A}(E_\nu,Q^2)"
    )

    st.subheader("Cortes (aceptación del muón)")
    st.markdown("Se aplican cortes cinemáticos equivalentes a la selección experimental:")
    st.latex(r"1.5~\mathrm{GeV}<E_\mu<20~\mathrm{GeV},\qquad \theta_\mu<20^\circ")


# -----------------------
# Tab 4
# -----------------------
with tabs[4]:
    st.subheader("Ajuste cuantitativo")
    q2_cent, q2_low, q2_high, data, model = compute_fluxfolded_prediction(MA, MV2, vector_ff, nQ2, Ev_max)
    chi2, chi2ndof = compute_chi2(data, model)

    c1, c2, c3 = st.columns(3)
    c1.metric("M_A [GeV]", f"{MA:.2f}")
    c2.metric("χ²", f"{chi2:.3f}")
    c3.metric("χ²/ndof", f"{chi2ndof:.3f}")

    st.markdown("Valor del ajuste (formato matemático):")
    st.latex(rf"\chi^2/\mathrm{{ndof}} = {chi2ndof:.3f}")

    st.markdown("Inspección bin a bin:")
    df = pd.DataFrame({
        "Q2low": q2_low,
        "Q2high": q2_high,
        "Q2cent": q2_cent,
        "data": data,
        "model": model,
        "ratio_data_model": data / np.where(np.abs(model) > 0, model, np.nan),
        "residual": data - model,
    })
    st.dataframe(df, use_container_width=True)

    if show_download:
        st.download_button(
            "Descargar tabla (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="minerva_hydrogen_comparison_bins.csv",
            mime="text/csv",
        )

    st.info(
        "Si el ratio en algún bin es muy grande, suele ser porque el modelo en ese bin cae mucho "
        "(flujo + cortes + cinemática). El impacto en χ² depende de las incertidumbres correlacionadas."
    )


# -----------------------
# Tab 5
# -----------------------
with tabs[5]:
    st.subheader("Simulador interactivo")
    q2_cent, q2_low, q2_high, data, model = compute_fluxfolded_prediction(MA, MV2, vector_ff, nQ2, Ev_max)
    ratio = data / np.where(np.abs(model) > 0, model, np.nan)

    left, right = st.columns(2)

    if PLOTLY_OK:
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=q2_cent, y=data, mode="markers", name="MINERvA data"))
            fig.add_trace(go.Scatter(x=q2_cent, y=model, mode="lines", name="Model (flux-folded + cuts)"))
            fig.update_layout(
                title="⟨dσ/dQ²⟩ vs Q²",
                xaxis_title="Q² [GeV²]",
                yaxis_title="⟨dσ/dQ²⟩ [10⁻³⁸ cm²/GeV²]",
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig2 = go.Figure()
            fig2.add_hline(y=1.0)
            fig2.add_trace(go.Scatter(x=q2_cent, y=ratio, mode="markers", name="Data/Model"))
            fig2.update_layout(
                title="Ratio: Data / Model",
                xaxis_title="Q² [GeV²]",
                yaxis_title="Data / Model",
                height=520,
            )
            if cap_ratio:
                fig2.update_yaxes(range=[0.0, ycap])
            st.plotly_chart(fig2, use_container_width=True)
    else:
        with left:
            fig, ax = plt.subplots()
            ax.plot(q2_cent, data, "o", label="MINERvA data")
            ax.plot(q2_cent, model, "-", label="Model")
            ax.set_xlabel(r"$Q^2\ [\mathrm{GeV}^2]$")
            ax.set_ylabel(r"$\langle d\sigma/dQ^2\rangle\ [10^{-38}\ \mathrm{cm^2/GeV^2}]$")
            ax.legend()
            st.pyplot(fig)

        with right:
            fig, ax = plt.subplots()
            ax.axhline(1.0)
            ax.plot(q2_cent, ratio, "o")
            ax.set_xlabel(r"$Q^2\ [\mathrm{GeV}^2]$")
            ax.set_ylabel("Data / Model")
            if cap_ratio:
                ax.set_ylim(0.0, ycap)
            st.pyplot(fig)

    st.caption("Consejo: cambia M_A y el modelo vectorial (Dipolo/GKeX) para ver cómo cambia la curva y χ²/ndof.")