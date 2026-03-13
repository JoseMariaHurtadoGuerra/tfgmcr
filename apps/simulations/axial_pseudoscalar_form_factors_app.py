"""
Standalone Streamlit page for the axial and pseudoscalar nucleon form factors.

Purpose
-------
This app is designed as a theory-heavy interactive module for a TFG on
charged-current neutrino scattering. It focuses on the nucleon axial sector:

    * axial form factor      G_A(Q^2)
    * pseudoscalar form factor G_P(Q^2)
    * combination             G_A'(Q^2) = G_A(Q^2) - tau G_P(Q^2)

The implementation follows the notation used in Guillermo D. Megias' thesis
and is intended to be either:

1) run directly as a standalone Streamlit app, or
2) imported into an existing multi-page Streamlit project and rendered by
   calling ``render_axial_form_factors_page()``.

Important integration note
--------------------------
If this file is imported into an existing Streamlit project whose main app
already calls ``st.set_page_config(...)``, keep that call ONLY in the main app.
This file calls ``st.set_page_config`` only inside ``main()`` so it remains
safe to import elsewhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Physical constants (natural units: hbar = c = 1)
# -----------------------------------------------------------------------------
G_A0 = -1.267          # axial coupling in the Q^2 -> 0 limit
M_A_STD = 1.032        # standard dipole axial mass in GeV
M_A_HIGH = 1.35        # higher effective value used in some nuclear fits
M_N = 0.939            # average nucleon mass in GeV
M_PI = 0.13957         # charged pion mass in GeV


@dataclass(frozen=True)
class CurveSpec:
    label: str
    mass: float
    linestyle: str
    linewidth: float = 2.2
    alpha: float = 1.0


# -----------------------------------------------------------------------------
# Physics model
# -----------------------------------------------------------------------------
def tau(q2: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(q2) / (4.0 * M_N**2)


def ga_dipole(q2: np.ndarray | float, m_a: float, g_a0: float = G_A0) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    return g_a0 / (1.0 + q2 / m_a**2) ** 2


def ga_monopole(q2: np.ndarray | float, m_a: float, g_a0: float = G_A0) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    return g_a0 / (1.0 + q2 / m_a**2)


def gp_pcac(q2: np.ndarray | float, ga_values: np.ndarray | float) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    ga_values = np.asarray(ga_values, dtype=float)
    return (4.0 * M_N**2 / (q2 + M_PI**2)) * ga_values


def ga_prime(q2: np.ndarray | float, ga_values: np.ndarray | float, gp_values: np.ndarray | float) -> np.ndarray:
    q2 = np.asarray(q2, dtype=float)
    return np.asarray(ga_values) - tau(q2) * np.asarray(gp_values)


def normalized_ga(ga_values: np.ndarray | float) -> np.ndarray:
    return np.asarray(ga_values) / G_A0


def normalized_gp(gp_values: np.ndarray | float) -> np.ndarray:
    # This reproduces the positive, dimensionless normalization used in the thesis-style plot.
    return np.asarray(gp_values) * M_PI**2 / (4.0 * M_N**2 * G_A0)


def normalized_gap(gap_values: np.ndarray | float) -> np.ndarray:
    return np.asarray(gap_values) / G_A0


# -----------------------------------------------------------------------------
# Numerical helpers
# -----------------------------------------------------------------------------
def safe_geomspace(q2_min: float, q2_max: float, n: int = 800) -> np.ndarray:
    q2_min = max(float(q2_min), 1.0e-6)
    q2_max = max(float(q2_max), q2_min * 1.001)
    return np.geomspace(q2_min, q2_max, n)


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return math.nan
    return 100.0 * (new - old) / old


def make_summary_table(q2_point: float, m_a_selected: float, m_a_reference: float, model_name: str) -> pd.DataFrame:
    if model_name == "Dipole":
        ga_fn = ga_dipole
    else:
        ga_fn = ga_monopole

    ga_sel = float(ga_fn(q2_point, m_a_selected))
    gp_sel = float(gp_pcac(q2_point, ga_sel))
    gap_sel = float(ga_prime(q2_point, ga_sel, gp_sel))

    ga_ref = float(ga_fn(q2_point, m_a_reference))
    gp_ref = float(gp_pcac(q2_point, ga_ref))
    gap_ref = float(ga_prime(q2_point, ga_ref, gp_ref))

    return pd.DataFrame(
        {
            "Quantity": [
                r"G_A / g_A",
                r"G_P m_pi^2 / [4 M_N^2 g_A]",
                r"G_A' / g_A",
                r"|G_A| / |G_A|_ref",
                r"|G_P| / |G_P|_ref",
                r"|G_A'| / |G_A'|_ref",
            ],
            "Selected": [
                normalized_ga(ga_sel),
                normalized_gp(gp_sel),
                normalized_gap(gap_sel),
                abs(ga_sel) / abs(ga_ref),
                abs(gp_sel) / abs(gp_ref),
                abs(gap_sel) / abs(gap_ref),
            ],
            "Reference": [
                normalized_ga(ga_ref),
                normalized_gp(gp_ref),
                normalized_gap(gap_ref),
                1.0,
                1.0,
                1.0,
            ],
        }
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _apply_common_axes_style(ax: plt.Axes, xscale: str = "log") -> None:
    ax.set_xscale(xscale)
    ax.grid(True, which="both", alpha=0.25)
    ax.tick_params(direction="in", top=True, right=True)


def plot_form_factors(
    q2: np.ndarray,
    curve_specs: list[CurveSpec],
    model_name: str,
    show_selected_q2: float | None = None,
) -> plt.Figure:
    ga_fn = ga_dipole if model_name == "Dipole" else ga_monopole

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    for spec in curve_specs:
        ga_vals = ga_fn(q2, spec.mass)
        gp_vals = gp_pcac(q2, ga_vals)

        ax1.plot(
            q2,
            normalized_ga(ga_vals),
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
            alpha=spec.alpha,
            label=spec.label,
        )
        ax2.plot(
            q2,
            normalized_gp(gp_vals),
            linestyle=spec.linestyle,
            linewidth=spec.linewidth,
            alpha=spec.alpha,
            label=spec.label,
        )

    if show_selected_q2 is not None:
        ax1.axvline(show_selected_q2, linestyle=":", linewidth=1.5)
        ax2.axvline(show_selected_q2, linestyle=":", linewidth=1.5)

    _apply_common_axes_style(ax1)
    _apply_common_axes_style(ax2)

    ax1.set_xlabel(r"$|Q^2|\; (\mathrm{GeV}^2)$")
    ax2.set_xlabel(r"$|Q^2|\; (\mathrm{GeV}^2)$")

    ax1.set_ylabel(r"$G_A/g_A$")
    ax2.set_ylabel(r"$G_P\,m_\pi^2 / \left[4M_N^2 g_A\right]$")

    ax1.set_title(r"Axial form factor")
    ax2.set_title(r"Pseudoscalar form factor")

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_sensitivity_ratios(
    q2: np.ndarray,
    m_a_selected: float,
    m_a_reference: float,
    model_name: str,
    show_selected_q2: float | None = None,
) -> plt.Figure:
    ga_fn = ga_dipole if model_name == "Dipole" else ga_monopole

    ga_sel = ga_fn(q2, m_a_selected)
    gp_sel = gp_pcac(q2, ga_sel)
    gap_sel = ga_prime(q2, ga_sel, gp_sel)

    ga_ref = ga_fn(q2, m_a_reference)
    gp_ref = gp_pcac(q2, ga_ref)
    gap_ref = ga_prime(q2, ga_ref, gp_ref)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    ax1.plot(q2, np.abs(ga_sel) / np.abs(ga_ref), linewidth=2.2, label=r"$|G_A|/|G_A|_{\rm ref}$")
    ax1.plot(q2, np.abs(gp_sel) / np.abs(gp_ref), linewidth=2.2, linestyle="--", label=r"$|G_P|/|G_P|_{\rm ref}$")

    ax2.plot(q2, np.abs(gap_sel) / np.abs(gap_ref), linewidth=2.2, label=r"$|G_A'|/|G_A'|_{\rm ref}$")
    ax2.plot(q2, (1.0 + tau(q2)) * (normalized_ga(ga_sel) ** 2) / ((1.0 + tau(q2)) * (normalized_ga(ga_ref) ** 2)),
             linewidth=2.2, linestyle="--", label=r"$R_T^{AA}/{R_T^{AA}}_{\rm ref}$")

    if show_selected_q2 is not None:
        ax1.axvline(show_selected_q2, linestyle=":", linewidth=1.5)
        ax2.axvline(show_selected_q2, linestyle=":", linewidth=1.5)

    _apply_common_axes_style(ax1)
    _apply_common_axes_style(ax2)

    ax1.set_xlabel(r"$|Q^2|\; (\mathrm{GeV}^2)$")
    ax2.set_xlabel(r"$|Q^2|\; (\mathrm{GeV}^2)$")
    ax1.set_ylabel("ratio to reference")
    ax2.set_ylabel("ratio to reference")
    ax1.set_title("Sensitivity of form factors to the axial mass")
    ax2.set_title("Sensitivity of derived axial ingredients")
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_axial_response_building_blocks(
    q2: np.ndarray,
    curve_specs: list[CurveSpec],
    model_name: str,
    show_selected_q2: float | None = None,
) -> plt.Figure:
    ga_fn = ga_dipole if model_name == "Dipole" else ga_monopole

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    for spec in curve_specs:
        ga_vals = ga_fn(q2, spec.mass)
        gp_vals = gp_pcac(q2, ga_vals)
        gap_vals = ga_prime(q2, ga_vals, gp_vals)

        # Dimensionless ingredients capturing the dependence of the axial responses.
        cc_like = normalized_gap(gap_vals) ** 2
        t_like = (1.0 + tau(q2)) * normalized_ga(ga_vals) ** 2

        ax1.plot(q2, normalized_gap(gap_vals), linestyle=spec.linestyle, linewidth=spec.linewidth,
                 alpha=spec.alpha, label=spec.label)
        ax2.plot(q2, cc_like, linestyle=spec.linestyle, linewidth=spec.linewidth,
                 alpha=spec.alpha, label=spec.label + "  (C/L)")
        ax2.plot(q2, t_like, linestyle=":", linewidth=max(spec.linewidth - 0.3, 1.2),
                 alpha=spec.alpha, label=spec.label + "  (T)")

    if show_selected_q2 is not None:
        ax1.axvline(show_selected_q2, linestyle=":", linewidth=1.5)
        ax2.axvline(show_selected_q2, linestyle=":", linewidth=1.5)

    _apply_common_axes_style(ax1)
    _apply_common_axes_style(ax2)

    ax1.set_xlabel(r"$|Q^2|\; (\mathrm{GeV}^2)$")
    ax2.set_xlabel(r"$|Q^2|\; (\mathrm{GeV}^2)$")
    ax1.set_ylabel(r"$G_A' / g_A$")
    ax2.set_ylabel("dimensionless response proxy")
    ax1.set_title(r"Combined axial quantity $G_A' = G_A - \tau G_P$")
    ax2.set_title(r"Longitudinal-like and transverse-like axial ingredients")
    ax1.legend(frameon=False)
    ax2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Streamlit content blocks
# -----------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .katex-display {
            overflow-x: auto;
            overflow-y: hidden;
            padding-bottom: 0.35rem;
        }
        .small-note {
            font-size: 0.95rem;
            opacity: 0.92;
        }
        .theory-card {
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 14px;
            padding: 1rem 1rem 0.85rem 1rem;
            margin-bottom: 0.75rem;
            background: rgba(250,250,250,0.02);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, body: str | None = None) -> None:
    st.markdown(f"## {title}")
    if body:
        st.markdown(body)


def render_context_tab() -> None:
    section_header(
        "Physical motivation",
        "This module is devoted to the **axial sector of the nucleon weak current**. The aim is to show, in a controlled and transparent way, how the axial mass modifies the shape of the axial form factor and how this propagates to the pseudoscalar factor and to the combinations that later enter charged-current quasielastic observables.",
    )

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown(
        "In charged-current neutrino scattering, the hadronic current contains **vector** and **axial** pieces. "
        "The vector sector can be related to the electromagnetic one through CVC, whereas the axial sector carries genuinely weak information and is therefore a privileged window into the internal spin-isospin structure of the nucleon."
    )
    st.latex(
        r"\widetilde\Gamma^{\mu} = \underbrace{F_1^V\gamma^{\mu} + \frac{iF_2^V}{2M_N}\sigma^{\mu\nu}Q_{\nu}}_{\text{vector}}"
        r" + \underbrace{G_A\gamma^{\mu}\gamma_5 + F_P Q^{\mu}\gamma_5}_{\text{axial + pseudoscalar}}"
    )
    st.markdown(
        "The app intentionally isolates the axial and pseudoscalar factors before moving to full cross sections. "
        "That makes the physics easier to read: first understand the **building blocks**, then understand the **observable**."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown(
        "The central parameter in this page is the **axial mass**. In the standard dipole ansatz it controls how quickly "
        r"$G_A(Q^2)$ falls with momentum transfer. A larger $M_A$ makes the falloff slower, so the form factor remains larger at moderate and high $Q^2$."
    )
    st.markdown(
        "This matters because the axial factor influences the size and shape of charged-current cross sections. "
        "At the same time, one must be careful with interpretation: in nuclear data, an effective increase of the fitted axial mass can sometimes mimic missing nuclear mechanisms rather than a genuine change in the nucleon structure itself."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.info(
        "What is simulated here is the **nucleon axial structure** and a few derived response ingredients. "
        "This is not yet a full neutrino-nucleus calculation with flux folding, binding, final-state interactions or 2p-2h MEC."
    )


def render_formalism_tab() -> None:
    section_header(
        "Formalism and notation",
        "The formulas below follow the notation typically used in the neutrino-scattering literature and match the structure discussed in Megias' work.",
    )

    st.markdown("### 1. Axial form factor")
    st.latex(r"G_A(Q^2) = \frac{g_A}{\left(1 + Q^2/M_A^2\right)^2}")
    st.markdown(
        "Here the normalization is fixed by the axial coupling in the elastic limit, whereas the parameter "
        r"$M_A$ controls the $Q^2$-dependence."
    )

    st.markdown("### 2. Pseudoscalar form factor from PCAC")
    st.latex(r"G_P(Q^2) = \frac{4M_N^2}{Q^2 + m_\pi^2}\, G_A(Q^2)")
    st.markdown(
        "This relation makes the pion pole explicit. Because of the factor "
        r"$1/(Q^2 + m_\pi^2)$, the pseudoscalar piece is strongly shaped by the low-$Q^2$ region and becomes much less relevant as $Q^2$ grows."
    )

    st.markdown("### 3. Useful dimensionless variable")
    st.latex(r"\tau = \frac{|Q^2|}{4M_N^2}")

    st.markdown("### 4. Combination that enters the axial longitudinal sector")
    st.latex(r"G_A'(Q^2) = G_A(Q^2) - \tau \, G_P(Q^2)")
    st.markdown(
        "This is a particularly useful object because the non-conservation of the axial current prevents one from collapsing the axial longitudinal channel into a single conserved-current expression, unlike what happens in the purely vector sector."
    )

    st.markdown("### 5. Single-nucleon axial ingredients")
    st.latex(r"R_{CC}^{AA} \propto \left[G_A'\right]^2, \qquad R_{CL}^{AA} \propto -\left[G_A'\right]^2, \qquad R_{LL}^{AA} \propto \left[G_A'\right]^2")
    st.latex(r"R_T^{AA} \propto 2(1+\tau)\, [G_A]^2")
    st.markdown(
        "So the simulator lets you inspect both the primary factors $G_A$ and $G_P$ and also the derived combination $G_A'$, which is the natural bridge toward the later response-function analysis."
    )

    with st.expander("Why does the pseudoscalar curve react less to changes in the axial mass?"):
        st.markdown(
            "Because its overall $Q^2$ behaviour is dominated by the pion-pole denominator. "
            "Even though $G_P$ is proportional to $G_A$, the extra low-scale factor involving the pion mass suppresses the visible sensitivity to $M_A$ much more strongly than in $G_A$ itself."
        )

    with st.expander("Why include a monopole option if the dipole one is the standard choice?"):
        st.markdown(
            "Because it is pedagogically useful to compare shapes. The literature commonly adopts a dipole ansatz, but comparing with a monopole profile helps the reader see that the issue is not only the value of the axial mass, but also the assumed functional form of the axial dependence on momentum transfer."
        )


def render_simulator_tab() -> None:
    st.markdown("### Controls")
    c1, c2, c3 = st.columns([1.1, 1.1, 1.2])

    with c1:
        model_name = st.radio("Axial ansatz", ["Dipole", "Monopole"], horizontal=True)
        m_a_selected = st.slider(r"Selected axial mass $M_A$ (GeV)", 0.75, 1.60, 1.032, 0.001)
        m_a_reference = st.slider(r"Reference axial mass $M_A^{\rm ref}$ (GeV)", 0.75, 1.60, 1.032, 0.001)

    with c2:
        q2_log_min, q2_log_max = st.slider(
            r"Plot range in $\\log_{10}(|Q^2|/\\mathrm{GeV}^2)$",
            min_value=-4.0,
            max_value=1.0,
            value=(-3.0, 1.0),
            step=0.1,
        )
        q2_probe_log = st.slider(
            r"Inspection point $\\log_{10}(|Q^2|/\\mathrm{GeV}^2)$",
            min_value=q2_log_min,
            max_value=q2_log_max,
            value=min(-1.0, q2_log_max),
            step=0.05,
        )
        q2_probe = 10.0 ** q2_probe_log

    with c3:
        show_standard = st.checkbox("Show standard benchmark $M_A = 1.032$ GeV", value=True)
        show_high = st.checkbox("Show high benchmark $M_A = 1.35$ GeV", value=True)
        show_reference = st.checkbox("Show custom reference curve", value=True)

    q2 = safe_geomspace(10.0 ** q2_log_min, 10.0 ** q2_log_max)

    curve_specs: list[CurveSpec] = [
        CurveSpec(label=fr"Selected  $M_A = {m_a_selected:.3f}$ GeV", mass=m_a_selected, linestyle="-", linewidth=2.6)
    ]
    if show_reference and abs(m_a_reference - m_a_selected) > 1e-12:
        curve_specs.append(CurveSpec(label=fr"Reference  $M_A = {m_a_reference:.3f}$ GeV", mass=m_a_reference, linestyle="--", linewidth=2.2, alpha=0.95))
    if show_standard and all(abs(spec.mass - M_A_STD) > 1e-12 for spec in curve_specs):
        curve_specs.append(CurveSpec(label=fr"Standard  $M_A = {M_A_STD:.3f}$ GeV", mass=M_A_STD, linestyle=":", linewidth=2.0, alpha=0.95))
    if show_high and all(abs(spec.mass - M_A_HIGH) > 1e-12 for spec in curve_specs):
        curve_specs.append(CurveSpec(label=fr"High  $M_A = {M_A_HIGH:.2f}$ GeV", mass=M_A_HIGH, linestyle="-.", linewidth=2.0, alpha=0.95))

    st.markdown("### Form factors")
    fig_ff = plot_form_factors(q2, curve_specs, model_name=model_name, show_selected_q2=q2_probe)
    st.pyplot(fig_ff, use_container_width=True)

    st.markdown("### Relative sensitivity to the axial mass")
    fig_ratio = plot_sensitivity_ratios(
        q2=q2,
        m_a_selected=m_a_selected,
        m_a_reference=m_a_reference,
        model_name=model_name,
        show_selected_q2=q2_probe,
    )
    st.pyplot(fig_ratio, use_container_width=True)

    st.markdown("### Derived axial quantity and response-oriented view")
    fig_gap = plot_axial_response_building_blocks(q2, curve_specs, model_name=model_name, show_selected_q2=q2_probe)
    st.pyplot(fig_gap, use_container_width=True)

    st.markdown("### Numerical readout at the selected momentum transfer")
    summary_df = make_summary_table(q2_probe, m_a_selected, m_a_reference, model_name)
    st.dataframe(summary_df.style.format({"Selected": "{:.6f}", "Reference": "{:.6f}"}), use_container_width=True)

    ga_fn = ga_dipole if model_name == "Dipole" else ga_monopole
    ga_sel = float(ga_fn(q2_probe, m_a_selected))
    gp_sel = float(gp_pcac(q2_probe, ga_sel))
    gap_sel = float(ga_prime(q2_probe, ga_sel, gp_sel))

    ga_ref = float(ga_fn(q2_probe, m_a_reference))
    gp_ref = float(gp_pcac(q2_probe, ga_ref))
    gap_ref = float(ga_prime(q2_probe, ga_ref, gp_ref))

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(
            label=r"Change in $|G_A|$ at selected $Q^2$",
            value=f"{abs(ga_sel)/abs(ga_ref):.4f}",
            delta=f"{pct_change(abs(ga_sel), abs(ga_ref)):+.2f}%",
        )
    with col_b:
        st.metric(
            label=r"Change in $|G_P|$ at selected $Q^2$",
            value=f"{abs(gp_sel)/abs(gp_ref):.4f}",
            delta=f"{pct_change(abs(gp_sel), abs(gp_ref)):+.2f}%",
        )
    with col_c:
        st.metric(
            label=r"Change in $|G_A'|$ at selected $Q^2$",
            value=f"{abs(gap_sel)/abs(gap_ref):.4f}",
            delta=f"{pct_change(abs(gap_sel), abs(gap_ref)):+.2f}%",
        )

    st.markdown(
        rf"At the inspection point $|Q^2| = {q2_probe:.4g}\,\\mathrm{{GeV}}^2$, the selected {model_name.lower()} ansatz with "
        rf"$M_A = {m_a_selected:.3f}\,\\mathrm{{GeV}}$ gives $G_A/g_A = {normalized_ga(ga_sel):.5f}$, "
        f"$G_P m_\\pi^2 / [4M_N^2 g_A] = {normalized_gp(gp_sel):.5f}$ and $G_A'/g_A = {normalized_gap(gap_sel):.5f}$."
    )

    with st.expander("How should these plots be read?"):
        st.markdown(
            "The first row shows the thesis-style normalized form factors. The second row answers a more practical question: "
            "**how much does the physics change relative to a reference choice of axial mass?** The third row moves one step closer "
            "to the response formalism by isolating the combination $G_A'$ and plotting simple dimensionless proxies of the longitudinal-like and transverse-like axial ingredients."
        )


def render_caveats_tab() -> None:
    section_header(
        "Phenomenology and caveats",
        "This is the place where the app should help the reader not over-interpret what they are seeing.",
    )

    st.warning(
        r"A fitted value like $M_A \simeq 1.35$ GeV in a nuclear analysis does **not automatically** imply that the fundamental nucleon axial form factor is different from the standard one. In many cases it may be acting as an effective parameter that compensates for missing nuclear strength."
    )

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown(
        "In quasielastic neutrino-nucleus scattering, the measured signal is never controlled only by the elementary nucleon form factors. "
        "Binding effects, Fermi motion, final-state interactions and multinucleon mechanisms can all reshape the observed cross section. "
        "For this reason, increasing the axial mass inside a simple one-body model can improve agreement with some data while actually hiding missing nuclear dynamics."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown(
        "The most important qualitative lessons of this page are the following:"
    )
    st.markdown(
        "1. **$G_A$ is the main driver of the visible axial-mass sensitivity** at moderate and high $Q^2$.  \n"
        "2. **$G_P$ is much less sensitive to $M_A$ in practice**, because the pion-pole denominator dominates its shape.  \n"
        "3. **The combination $G_A'$ matters for the axial charge/longitudinal sector**, so the app includes it explicitly.  \n"
        "4. **A full cross-section interpretation requires nuclear dynamics**, which lies beyond the scope of this page but motivates the next layers of the TFG simulators."
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_references_tab() -> None:
    section_header(
        "Guide references for this page",
        "This tab is not meant as a formal bibliography manager but as a compact reading map for the reader of the app.",
    )

    st.markdown(
        "- G. D. Megias, *Doctoral Thesis* — especially the discussion of the weak hadronic current, axial and pseudoscalar form factors, and the appendix on monopole versus dipole axial ansätze.  \n"
        "- G. D. Megias, *Master thesis / first chapters* — introductory discussion of the hadronic structure and pedagogical plots of the axial and pseudoscalar factors.  \n"
        "- C. H. Llewellyn Smith, *Neutrino Reactions at Accelerator Energies* — classic reference for neutrino-nucleon weak form factors and cross sections.  \n"
        "- Walecka, *Electron Scattering for Nuclear and Nucleon Structure* — especially useful for separating current structure, response functions and form-factor language.  \n"
        "- Formaggio and Zeller, *From eV to EeV: Neutrino Cross Sections Across Energy Scales* — broad modern review that situates CCQE physics inside the larger neutrino-interaction landscape."
    )

    st.caption(
        "Recommended reading order inside the app: Motivation -> Formalism -> Simulator -> Caveats."
    )


def render_axial_form_factors_page() -> None:
    inject_css()

    st.title("Axial and pseudoscalar nucleon form factors")
    st.markdown(
        "Interactive theory page for the weak axial structure of the nucleon, with direct emphasis on the role of the axial mass in "
        r"$G_A(Q^2)$, $G_P(Q^2)$ and in the derived combination $G_A'(Q^2) = G_A - \tau G_P$."
    )

    st.markdown(
        "This page is designed as a bridge between the formal definition of the weak hadronic current and the later analysis of charged-current observables. "
        "It first isolates the nucleon-level ingredients and only then connects them with the longitudinal/transverse response language."
    )

    tabs = st.tabs(
        [
            "Physical context",
            "Formalism",
            "Interactive simulator",
            "Phenomenology & caveats",
            "Reading map",
        ]
    )

    with tabs[0]:
        render_context_tab()
    with tabs[1]:
        render_formalism_tab()
    with tabs[2]:
        render_simulator_tab()
    with tabs[3]:
        render_caveats_tab()
    with tabs[4]:
        render_references_tab()


def main() -> None:
    st.set_page_config(page_title="Axial nucleon form factors", layout="wide")
    render_axial_form_factors_page()


if __name__ == "__main__":
    main()
