\
"""
Página independiente de Streamlit para los factores de forma axial y pseudoescalar del nucleón.

Propósito
---------
Esta app está diseñada como un módulo interactivo con abundante contenido teórico
para un TFG sobre dispersión de neutrinos por corriente cargada. Se centra en el
sector axial del nucleón:

    * factor de forma axial          G_A(Q^2)
    * factor de forma pseudoescalar  G_P(Q^2)
    * combinación                    G_A'(Q^2) = G_A(Q^2) - tau G_P(Q^2)

La implementación sigue la notación utilizada en la tesis de Guillermo D. Megías
y está pensada para:

1) ejecutarse directamente como app independiente de Streamlit, o bien
2) importarse en un proyecto existente de Streamlit con varias páginas y renderizarse
   llamando a ``render_axial_form_factors_page()``.

Nota importante de integración
------------------------------
Si este archivo se importa en un proyecto de Streamlit existente cuya app principal
ya llama a ``st.set_page_config(...)``, mantén esa llamada SOLO en la app principal.
Este archivo llama a ``st.set_page_config`` únicamente dentro de ``main()``, de modo
que sigue siendo seguro importarlo en otro lugar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Constantes físicas (unidades naturales: hbar = c = 1)
# -----------------------------------------------------------------------------
G_A0 = -1.267          # acoplamiento axial en el límite Q^2 -> 0
M_A_STD = 1.032        # masa axial dipolar estándar en GeV
M_A_HIGH = 1.35        # valor efectivo más alto usado en algunos ajustes nucleares
M_N = 0.939            # masa media del nucleón en GeV
M_PI = 0.13957         # masa del pión cargado en GeV


@dataclass(frozen=True)
class CurveSpec:
    label: str
    mass: float
    linestyle: str
    linewidth: float = 2.2
    alpha: float = 1.0


# -----------------------------------------------------------------------------
# Modelo físico
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
    # Esto reproduce la normalización positiva y adimensional usada en la figura con estilo de tesis.
    return np.asarray(gp_values) * M_PI**2 / (4.0 * M_N**2 * G_A0)


def normalized_gap(gap_values: np.ndarray | float) -> np.ndarray:
    return np.asarray(gap_values) / G_A0


# -----------------------------------------------------------------------------
# Utilidades numéricas
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
    if model_name == "Dipolar":
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
            "Magnitud": [
                r"G_A / g_A",
                r"G_P m_pi^2 / [4 M_N^2 g_A]",
                r"G_A' / g_A",
                r"|G_A| / |G_A|_ref",
                r"|G_P| / |G_P|_ref",
                r"|G_A'| / |G_A'|_ref",
            ],
            "Seleccionado": [
                normalized_ga(ga_sel),
                normalized_gp(gp_sel),
                normalized_gap(gap_sel),
                abs(ga_sel) / abs(ga_ref),
                abs(gp_sel) / abs(gp_ref),
                abs(gap_sel) / abs(gap_ref),
            ],
            "Referencia": [
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
# Gráficas
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
    ga_fn = ga_dipole if model_name == "Dipolar" else ga_monopole

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

    ax1.set_title(r"Factor de forma axial")
    ax2.set_title(r"Factor de forma pseudoescalar")

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
    ga_fn = ga_dipole if model_name == "Dipolar" else ga_monopole

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
    ax1.set_ylabel("cociente respecto a la referencia")
    ax2.set_ylabel("cociente respecto a la referencia")
    ax1.set_title("Sensibilidad de los factores de forma a la masa axial")
    ax2.set_title("Sensibilidad de las magnitudes axiales derivadas")

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
    ga_fn = ga_dipole if model_name == "Dipolar" else ga_monopole

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    ax1, ax2 = axes

    for spec in curve_specs:
        ga_vals = ga_fn(q2, spec.mass)
        gp_vals = gp_pcac(q2, ga_vals)
        gap_vals = ga_prime(q2, ga_vals, gp_vals)

        # Ingredientes adimensionales que capturan la dependencia de las respuestas axiales.
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
    ax2.set_ylabel("proxy adimensional de respuesta")
    ax1.set_title(r"Magnitud axial combinada $G_A' = G_A - \tau G_P$")
    ax2.set_title(r"Ingredientes axiales tipo longitudinal y transversal")
    ax1.legend(frameon=False)
    ax2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Bloques de contenido Streamlit
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
        "Motivación física",
        "Este módulo está dedicado al **sector axial de la corriente débil del nucleón**. El objetivo es mostrar, de forma controlada y transparente, cómo la masa axial modifica la forma del factor de forma axial y cómo eso se propaga al factor pseudoescalar y a las combinaciones que más tarde entran en los observables cuasielásticos por corriente cargada.",
    )

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown(
        "En la dispersión de neutrinos por corriente cargada, la corriente hadrónica contiene partes **vectorial** y **axial**. "
        "El sector vectorial puede relacionarse con el electromagnético mediante CVC, mientras que el sector axial contiene información genuinamente débil y constituye, por tanto, una ventana privilegiada a la estructura interna spin-isospín del nucleón."
    )
    st.latex(
        r"\widetilde\Gamma^{\mu} = \underbrace{F_1^V\gamma^{\mu} + \frac{iF_2^V}{2M_N}\sigma^{\mu\nu}Q_{\nu}}_{\text{vectorial}}"
        r" + \underbrace{G_A\gamma^{\mu}\gamma_5 + F_P Q^{\mu}\gamma_5}_{\text{axial + pseudoescalar}}"
    )
    st.markdown(
        "La app aísla intencionadamente los factores axial y pseudoescalar antes de pasar a las secciones eficaces completas. "
        "Eso hace la física más legible: primero entender los **bloques fundamentales**, y después entender el **observable**."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown(
        "El parámetro central de esta página es la **masa axial**. En el ansatz dipolar estándar controla la rapidez con la que "
        r"$G_A(Q^2)$ cae con el momento transferido. Un $M_A$ mayor hace que la caída sea más lenta, de modo que el factor de forma permanece más grande a $Q^2$ moderados y altos."
    )
    st.markdown(
        "Esto importa porque el factor axial influye en el tamaño y la forma de las secciones eficaces por corriente cargada. "
        "Al mismo tiempo, hay que ser cuidadosos con la interpretación: en datos nucleares, un aumento efectivo de la masa axial ajustada puede en ocasiones estar imitando mecanismos nucleares ausentes, más que un cambio genuino en la propia estructura del nucleón."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.info(
        "Lo que aquí se simula es la **estructura axial del nucleón** y algunos ingredientes derivados de las respuestas. "
        "Todavía no se trata de un cálculo completo neutrino-núcleo con folding en flujo, ligadura, interacciones en el estado final o procesos 2p-2h MEC."
    )


def render_formalism_tab() -> None:
    section_header(
        "Formalismo y notación",
        "Las fórmulas siguientes siguen la notación típica de la literatura de dispersión de neutrinos y reproducen la estructura discutida en el trabajo de Megías.",
    )

    st.markdown("### 1. Factor de forma axial")
    st.latex(r"G_A(Q^2) = \frac{g_A}{\left(1 + Q^2/M_A^2\right)^2}")
    st.markdown(
        "Aquí la normalización queda fijada por el acoplamiento axial en el límite elástico, mientras que el parámetro "
        r"$M_A$ controla la dependencia en $Q^2$."
    )

    st.markdown("### 2. Factor de forma pseudoescalar a partir de PCAC")
    st.latex(r"G_P(Q^2) = \frac{4M_N^2}{Q^2 + m_\pi^2}\, G_A(Q^2)")
    st.markdown(
        "Esta relación hace explícito el polo del pión. Debido al factor "
        r"$1/(Q^2 + m_\pi^2)$, la parte pseudoescalar queda fuertemente moldeada por la región de bajo $Q^2$ y se vuelve mucho menos relevante conforme crece $Q^2$."
    )

    st.markdown("### 3. Variable adimensional útil")
    st.latex(r"\tau = \frac{|Q^2|}{4M_N^2}")

    st.markdown("### 4. Combinación que entra en el sector axial longitudinal")
    st.latex(r"G_A'(Q^2) = G_A(Q^2) - \tau \, G_P(Q^2)")
    st.markdown(
        "Esta es una magnitud especialmente útil porque la no conservación de la corriente axial impide reducir el canal axial longitudinal a una única expresión de corriente conservada, a diferencia de lo que ocurre en el sector puramente vectorial."
    )

    st.markdown("### 5. Ingredientes axiales de nucleón individual")
    st.latex(r"R_{CC}^{AA} \propto \left[G_A'\right]^2, \qquad R_{CL}^{AA} \propto -\left[G_A'\right]^2, \qquad R_{LL}^{AA} \propto \left[G_A'\right]^2")
    st.latex(r"R_T^{AA} \propto 2(1+\tau)\, [G_A]^2")
    st.markdown(
        "Por eso el simulador permite inspeccionar tanto los factores primarios $G_A$ y $G_P$ como la combinación derivada $G_A'$, que constituye el puente natural hacia el análisis posterior en términos de funciones de respuesta."
    )

    with st.expander("¿Por qué la curva pseudoescalar reacciona menos a cambios en la masa axial?"):
        st.markdown(
            "Porque su comportamiento global en $Q^2$ está dominado por el denominador del polo del pión. "
            "Aunque $G_P$ es proporcional a $G_A$, el factor extra de baja escala que involucra la masa del pión suprime mucho más la sensibilidad visible a $M_A$ que en el propio $G_A$."
        )

    with st.expander("¿Por qué incluir una opción monopolar si la dipolar es la elección estándar?"):
        st.markdown(
            "Porque es pedagógicamente útil comparar formas funcionales. La literatura adopta habitualmente un ansatz dipolar, pero contrastarlo con un perfil monopolar ayuda al lector a ver que la cuestión no es solo el valor de la masa axial, sino también la forma funcional asumida para la dependencia axial con el momento transferido."
        )


def render_simulator_tab() -> None:
    st.markdown("### Controles")
    c1, c2, c3 = st.columns([1.1, 1.1, 1.2])

    with c1:
        model_name = st.radio("Ansatz axial", ["Dipolar", "Monopolar"], horizontal=True)
        m_a_selected = st.slider(r"Masa axial seleccionada $M_A$ (GeV)", 0.75, 1.60, 1.032, 0.001)
        m_a_reference = st.slider(r"Masa axial de referencia $M_A^{\rm ref}$ (GeV)", 0.75, 1.60, 1.032, 0.001)

    with c2:
        q2_log_min, q2_log_max = st.slider(
            r"Rango de representación en $\\log_{10}(|Q^2|/\\mathrm{GeV}^2)$",
            min_value=-4.0,
            max_value=1.0,
            value=(-3.0, 1.0),
            step=0.1,
        )
        q2_probe_log = st.slider(
            r"Punto de inspección $\\log_{10}(|Q^2|/\\mathrm{GeV}^2)$",
            min_value=q2_log_min,
            max_value=q2_log_max,
            value=min(-1.0, q2_log_max),
            step=0.05,
        )
        q2_probe = 10.0 ** q2_probe_log

    with c3:
        show_standard = st.checkbox("Mostrar benchmark estándar $M_A = 1.032$ GeV", value=True)
        show_high = st.checkbox("Mostrar benchmark alto $M_A = 1.35$ GeV", value=True)
        show_reference = st.checkbox("Mostrar curva de referencia personalizada", value=True)

    q2 = safe_geomspace(10.0 ** q2_log_min, 10.0 ** q2_log_max)

    curve_specs: list[CurveSpec] = [
        CurveSpec(label=fr"Seleccionada  $M_A = {m_a_selected:.3f}$ GeV", mass=m_a_selected, linestyle="-", linewidth=2.6)
    ]
    if show_reference and abs(m_a_reference - m_a_selected) > 1e-12:
        curve_specs.append(CurveSpec(label=fr"Referencia  $M_A = {m_a_reference:.3f}$ GeV", mass=m_a_reference, linestyle="--", linewidth=2.2, alpha=0.95))
    if show_standard and all(abs(spec.mass - M_A_STD) > 1e-12 for spec in curve_specs):
        curve_specs.append(CurveSpec(label=fr"Estándar  $M_A = {M_A_STD:.3f}$ GeV", mass=M_A_STD, linestyle=":", linewidth=2.0, alpha=0.95))
    if show_high and all(abs(spec.mass - M_A_HIGH) > 1e-12 for spec in curve_specs):
        curve_specs.append(CurveSpec(label=fr"Alto  $M_A = {M_A_HIGH:.2f}$ GeV", mass=M_A_HIGH, linestyle="-.", linewidth=2.0, alpha=0.95))

    st.markdown("### Factores de forma")
    fig_ff = plot_form_factors(q2, curve_specs, model_name=model_name, show_selected_q2=q2_probe)
    st.pyplot(fig_ff, use_container_width=True)

    st.markdown("### Sensibilidad relativa a la masa axial")
    fig_ratio = plot_sensitivity_ratios(
        q2=q2,
        m_a_selected=m_a_selected,
        m_a_reference=m_a_reference,
        model_name=model_name,
        show_selected_q2=q2_probe,
    )
    st.pyplot(fig_ratio, use_container_width=True)

    st.markdown("### Magnitud axial derivada y lectura orientada a respuestas")
    fig_gap = plot_axial_response_building_blocks(q2, curve_specs, model_name=model_name, show_selected_q2=q2_probe)
    st.pyplot(fig_gap, use_container_width=True)

    st.markdown("### Lectura numérica en el momento transferido seleccionado")
    summary_df = make_summary_table(q2_probe, m_a_selected, m_a_reference, model_name)
    st.dataframe(summary_df.style.format({"Seleccionado": "{:.6f}", "Referencia": "{:.6f}"}), use_container_width=True)

    ga_fn = ga_dipole if model_name == "Dipolar" else ga_monopole
    ga_sel = float(ga_fn(q2_probe, m_a_selected))
    gp_sel = float(gp_pcac(q2_probe, ga_sel))
    gap_sel = float(ga_prime(q2_probe, ga_sel, gp_sel))

    ga_ref = float(ga_fn(q2_probe, m_a_reference))
    gp_ref = float(gp_pcac(q2_probe, ga_ref))
    gap_ref = float(ga_prime(q2_probe, ga_ref, gp_ref))

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(
            label=r"Cambio en $|G_A|$ en el $Q^2$ seleccionado",
            value=f"{abs(ga_sel)/abs(ga_ref):.4f}",
            delta=f"{pct_change(abs(ga_sel), abs(ga_ref)):+.2f}%",
        )
    with col_b:
        st.metric(
            label=r"Cambio en $|G_P|$ en el $Q^2$ seleccionado",
            value=f"{abs(gp_sel)/abs(gp_ref):.4f}",
            delta=f"{pct_change(abs(gp_sel), abs(gp_ref)):+.2f}%",
        )
    with col_c:
        st.metric(
            label=r"Cambio en $|G_A'|$ en el $Q^2$ seleccionado",
            value=f"{abs(gap_sel)/abs(gap_ref):.4f}",
            delta=f"{pct_change(abs(gap_sel), abs(gap_ref)):+.2f}%",
        )

    st.markdown(
        rf"En el punto de inspección $|Q^2| = {q2_probe:.4g}\,\mathrm{{GeV}}^2$, el ansatz {model_name.lower()} seleccionado con "
        rf"$M_A = {m_a_selected:.3f}\,\mathrm{{GeV}}$ da lugar a $G_A/g_A = {normalized_ga(ga_sel):.5f}$, "
        rf"$G_P m_\pi^2 / [4M_N^2 g_A] = {normalized_gp(gp_sel):.5f}$ y $G_A'/g_A = {normalized_gap(gap_sel):.5f}$."
    )

    with st.expander("¿Cómo deben leerse estas gráficas?"):
        st.markdown(
            "La primera fila muestra los factores de forma normalizados con el estilo de la tesis. La segunda fila responde a una pregunta más práctica: "
            "**¿cuánto cambia la física respecto a una elección de referencia de la masa axial?** La tercera fila da un paso más hacia el formalismo de respuestas, aislando la combinación $G_A'$ y representando proxies adimensionales sencillos de los ingredientes axiales de tipo longitudinal y transversal."
        )


def render_caveats_tab() -> None:
    section_header(
        "Fenomenología y advertencias",
        "Aquí es donde la app debe ayudar al lector a no sobreinterpretar lo que está viendo.",
    )

    st.warning(
        r"Un valor ajustado como $M_A \simeq 1.35$ GeV en un análisis nuclear **no implica automáticamente** que el factor de forma axial fundamental del nucleón sea distinto del estándar. En muchos casos puede estar actuando como un parámetro efectivo que compensa fuerza nuclear ausente en el modelo."
    )

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown(
        "En la dispersión cuasielástica de neutrinos sobre núcleos, la señal medida nunca está controlada únicamente por los factores de forma elementales del nucleón. "
        "Los efectos de ligadura, el movimiento de Fermi, las interacciones en el estado final y los mecanismos multinucleónicos pueden remodelar la sección eficaz observada. "
        "Por ello, aumentar la masa axial dentro de un modelo simple de un cuerpo puede mejorar el acuerdo con algunos datos y, sin embargo, estar ocultando dinámica nuclear ausente."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="theory-card">', unsafe_allow_html=True)
    st.markdown(
        "Las lecciones cualitativas más importantes de esta página son las siguientes:"
    )
    st.markdown(
        "1. **$G_A$ es el principal responsable de la sensibilidad visible a la masa axial** a $Q^2$ moderados y altos.  \n"
        "2. **$G_P$ es, en la práctica, mucho menos sensible a $M_A$**, porque el denominador del polo del pión domina su forma.  \n"
        "3. **La combinación $G_A'$ importa para el sector axial carga/longitudinal**, así que la app la incluye explícitamente.  \n"
        "4. **Una interpretación completa de la sección eficaz requiere dinámica nuclear**, algo que queda fuera del alcance de esta página pero motiva las siguientes capas de simuladores del TFG."
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_references_tab() -> None:
    section_header(
        "Referencias guía de esta página",
        "Esta pestaña no pretende ser un gestor bibliográfico formal, sino una hoja de ruta compacta de lectura para quien use la app.",
    )

    st.markdown(
        "- G. D. Megías, *Tesis doctoral* — especialmente la discusión de la corriente hadrónica débil, los factores de forma axial y pseudoescalar, y el apéndice sobre ansätze axiales monopolar frente a dipolar.  \n"
        "- G. D. Megías, *Master thesis / first chapters* — discusión introductoria de la estructura hadrónica y gráficas pedagógicas de los factores axial y pseudoescalar.  \n"
        "- C. H. Llewellyn Smith, *Neutrino Reactions at Accelerator Energies* — referencia clásica sobre factores de forma débiles y secciones eficaces neutrino-nucleón.  \n"
        "- Walecka, *Electron Scattering for Nuclear and Nucleon Structure* — especialmente útil para separar estructura de corrientes, funciones de respuesta y lenguaje de factores de forma.  \n"
        "- Formaggio and Zeller, *From eV to EeV: Neutrino Cross Sections Across Energy Scales* — revisión moderna amplia que sitúa la física CCQE dentro del panorama general de interacciones de neutrinos."
    )

    


def render_axial_form_factors_page() -> None:
    inject_css()

    st.title("Factores de forma axial y pseudoescalar del nucleón")
    st.markdown(
        "Página teórica interactiva dedicada a la estructura axial débil del nucleón, con énfasis directo en el papel de la masa axial en "
        r"$G_A(Q^2)$, $G_P(Q^2)$ y en la combinación derivada $G_A'(Q^2) = G_A - \tau G_P$."
    )

    st.markdown(
        "Esta página está diseñada como un puente entre la definición formal de la corriente hadrónica débil y el análisis posterior de observables por corriente cargada. "
        "Primero aísla los ingredientes a nivel de nucleón y solo después los conecta con el lenguaje de respuestas longitudinales/transversales."
    )

    tabs = st.tabs(
        [
            "Contexto físico",
            "Formalismo",
            "Simulador interactivo",
            "Fenomenología y advertencias",
            "Hoja de lectura",
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
    st.set_page_config(page_title="Factores de forma axiales del nucleón", layout="wide")
    render_axial_form_factors_page()


if __name__ == "__main__":
    main()
