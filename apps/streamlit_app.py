# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:05:06 2026

@author: User
"""

# apps/streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.ccqe_curves import curve_theta, curve_q2_reparam

st.set_page_config(page_title="CCQE Explorer", layout="wide")

st.title("CCQE explorer: dσ/dΩ vs θμ y vs |Q²|")

with st.sidebar:
    Ev = st.slider("Eν [GeV]", min_value=0.2, max_value=3.0, value=1.0, step=0.05)
    vector_model = st.selectbox("Vector FF model", ["galster", "gkex"], index=1)
    MA = st.selectbox("M_A [GeV]", [1.03, 1.35], index=0)
    is_antinu = st.checkbox("Antineutrino (ν̄)", value=False)

col1, col2 = st.columns(2)

# --- θ plot ---
theta_deg, dsdo_theta = curve_theta(Ev, vector_model=vector_model, MA=MA, is_antinu=is_antinu)

with col2:
    st.subheader("dσ/dΩ vs θμ")
    fig = plt.figure()
    plt.plot(theta_deg, dsdo_theta)
    plt.xlabel("θμ [deg]")
    plt.ylabel("dσ/dΩ [cm²/sr]")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    df_theta = pd.DataFrame({"theta_deg": theta_deg, "dsigma_dOmega_cm2_sr": dsdo_theta})
    st.download_button(
        "Download θ curve (CSV)",
        df_theta.to_csv(index=False).encode("utf-8"),
        file_name="ccqe_theta_curve.csv",
        mime="text/csv",
    )

# --- Q2 plot ---
Q2, dsdo_q2 = curve_q2_reparam(Ev, vector_model=vector_model, MA=MA, is_antinu=is_antinu)

with col1:
    st.subheader("dσ/dΩ vs |Q²|  (reparametrized)")
    fig = plt.figure()
    plt.plot(Q2, dsdo_q2)
    plt.xlabel("|Q²| [GeV²]")
    plt.ylabel("dσ/dΩ [cm²/sr]")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    df_q2 = pd.DataFrame({"Q2_GeV2": Q2, "dsigma_dOmega_cm2_sr": dsdo_q2})
    st.download_button(
        "Download Q² curve (CSV)",
        df_q2.to_csv(index=False).encode("utf-8"),
        file_name="ccqe_q2_curve.csv",
        mime="text/csv",
    )

st.caption("Models: Galster vs GKex (Lomon). QE kinematics. Units: natural units internally; output in cm²/sr.")