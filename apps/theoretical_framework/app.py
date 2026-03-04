
# apps/theoretical_framework/app.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

import streamlit as st

# ------------------------------------------------------------
# Robust local import (appendix.py in the same folder)
# ------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from appendix import render_appendix  # noqa: E402

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Theoretical Framework — CC ν–N",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Global CSS:
# - Better typography
# - KaTeX display overflow protection (long equations won't break layout)
# ------------------------------------------------------------
st.markdown(
    r"""
<style>
.block-container {
  padding-top: 2.0rem;
  padding-bottom: 2.0rem;
  max-width: 1200px;
}
section[data-testid="stSidebar"] .block-container {
  padding-top: 1.25rem;
}
.katex-display {
  overflow-x: auto;
  overflow-y: hidden;
  padding-bottom: 0.25rem;
}
h1, h2, h3 { margin-bottom: 0.6rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def p(text: str) -> None:
    """Paragraph (markdown). Use $...$ for inline math. Prefer raw strings r'...'."""
    st.markdown(text)

def eq(latex: str) -> None:
    """Displayed equation (robust KaTeX rendering)."""
    st.latex(latex)

def section(title: str, subtitle: str | None = None) -> None:
    st.header(title)
    if subtitle:
        st.caption(subtitle)


# ------------------------------------------------------------
# Chapters (formal style, equation-by-equation)
# ------------------------------------------------------------
def render_home() -> None:
    section(
        "Theoretical Framework",
        "Interacción neutrino–nucleón por corrientes cargadas (CC) — presentación autocontenida",
    )
    p(r"""
Este marco teórico expone, de forma autocontenida y con notación consistente, los elementos necesarios para describir
la interacción **neutrino–nucleón** mediante **corrientes cargadas (CC)**.

Los cálculos extensos de álgebra de Dirac (típicamente **trazas** y **contracciones**) se recogen en el **Apéndice**,
manteniendo en el cuerpo principal una lectura clara.
""")
    p("Procesos prototipo (CCQE libre):")
    eq(r"\nu_\ell(k)+n(p_i)\to \ell^-(k')+p(p_f),\qquad \bar\nu_\ell(k)+p(p_i)\to \ell^+(k')+n(p_f).")
    p("Observables diferenciales de interés:")
    eq(r"\frac{d\sigma}{d|Q^2|},\qquad \frac{d\sigma}{d\theta_\mu}\quad (\text{o } d\sigma/d\cos\theta_\mu).")


def render_ch1() -> None:
    section("1. Convenciones y notación", "Definiciones y convenciones utilizadas en todo el documento")

    p("Unidades naturales:")
    eq(r"\hbar=c=1.")

    p("Métrica y producto escalar:")
    eq(r"g^{\mu\nu}=\mathrm{diag}(+1,-1,-1,-1),\qquad a\cdot b=a^0b^0-\mathbf{a}\cdot\mathbf{b}.")

    p("Cuadrimomentos:")
    eq(r"k^\mu=(E_\nu,\mathbf{k}),\quad k'^\mu=(E_\ell,\mathbf{k'}),\quad p_i^\mu=(E_i,\mathbf{p_i}),\quad p_f^\mu=(E_f,\mathbf{p_f}).")

    p("En el laboratorio (nucleón inicial en reposo):")
    eq(r"p_i^\mu=(M,\mathbf{0}).")

    p("Transferencia de cuatro-momento:")
    eq(r"Q^\mu\equiv k^\mu-k'^\mu=(\omega,\mathbf{q}).")
    eq(r"Q^2=\omega^2-\mathbf{q}^2\le 0,\qquad |Q^2|\equiv -Q^2\ge 0.")

    p("Álgebra de Dirac (mínimo imprescindible):")
    eq(r"\{\gamma^\mu,\gamma^\nu\}=2g^{\mu\nu},\qquad \not{a}\equiv \gamma_\mu a^\mu.")
    eq(r"\sigma^{\mu\nu}\equiv \frac{i}{2}[\gamma^\mu,\gamma^\nu],\qquad \gamma^5\equiv i\gamma^0\gamma^1\gamma^2\gamma^3.")
    eq(r"P_L=\frac{1-\gamma^5}{2},\qquad P_R=\frac{1+\gamma^5}{2}.")

    p("Convención de Levi–Civita:")
    eq(r"\epsilon^{0123}=+1.")

    st.info("Las identidades completas, trazas y contracciones extensas se recopilan en el **Apéndice**.")


def render_ch2() -> None:
    section("2. Cinemática relativista (laboratorio)", r"Relaciones entre $E_\nu$, $E_\ell$, $\theta_\ell$ y $|Q^2|$")

    p("Definiciones:")
    eq(r"k^\mu=(E_\nu,\mathbf{k}),\qquad k'^\mu=(E_\ell,\mathbf{k'}),\qquad p_\ell\equiv |\mathbf{k'}|=\sqrt{E_\ell^2-m_\ell^2}.")
    eq(r"k^2=0,\qquad |\mathbf{k}|=E_\nu.")

    p(r"Definimos $\theta_\ell$ como el ángulo entre $\mathbf{k}$ y $\mathbf{k'}$:")
    eq(r"\mathbf{k}\cdot\mathbf{k'}=E_\nu\,p_\ell\cos\theta_\ell.")

    p(r"Transferencia $Q^\mu=k^\mu-k'^\mu$ y su invariante:")
    eq(r"Q^2=(k-k')^2=k^2+k'^2-2k\cdot k'=m_\ell^2-2k\cdot k'.")
    eq(r"k\cdot k'=E_\nu E_\ell - E_\nu p_\ell\cos\theta_\ell.")
    eq(r"Q^2=m_\ell^2-2E_\nu(E_\ell-p_\ell\cos\theta_\ell),\qquad |Q^2|=2E_\nu(E_\ell-p_\ell\cos\theta_\ell)-m_\ell^2.")

    p("Relación con $(\omega,\mathbf{q})$:")
    eq(r"Q^\mu=(\omega,\mathbf{q}),\qquad \omega=E_\nu-E_\ell,\qquad |Q^2|=\mathbf{q}^2-\omega^2.")

    p("Caso cuasi-elástico libre: $W^2=(p_i+Q)^2=M^2$ con $p_i=(M,\mathbf{0})$.")
    eq(r"W^2=M^2+2M\omega+Q^2.")
    eq(r"W^2=M^2\ \Rightarrow\ 2M\omega+Q^2=0\ \Rightarrow\ \omega=\frac{|Q^2|}{2M}.")

    p("Fórmula de reconstrucción (útil en el laboratorio):")
    eq(r"E_\nu=\frac{M E_\ell-\frac{m_\ell^2}{2}}{\,M - E_\ell + p_\ell\cos\theta_\ell\,}.")


def render_ch3() -> None:
    section("3. Interacción débil CC y amplitud", "Límite de Fermi y estructura V–A")

    p(r"En el régimen $|Q^2|\ll M_W^2$ se emplea la interacción efectiva local tipo Fermi:")
    eq(r"""
\mathcal{L}_{\mathrm{CC}}^{\mathrm{eff}}
=
-\frac{G_F\cos\theta_C}{\sqrt{2}}\;
\Big[\bar{\ell}\gamma_\mu(1-\gamma^5)\nu_\ell\Big]\;
\Big[\bar{N}'\,\Gamma^\mu(Q^2)\,N\Big]
+\mathrm{h.c.}
""")

    p("Para CCQE (nucleón libre), la amplitud invariante es:")
    eq(r"""
\mathcal{M}
=
\frac{G_F\cos\theta_C}{\sqrt{2}}
\Big[\bar u(k')\,\gamma_\mu(1-\gamma^5)\,u(k)\Big]
\Big[\bar u(p_f)\,\Gamma^\mu(Q^2)\,u(p_i)\Big].
""")

    p("Definimos las corrientes leptónica y hadrónica:")
    eq(r"L_\mu \equiv \bar u(k')\,\gamma_\mu(1-\gamma^5)\,u(k),\qquad J^\mu \equiv \bar u(p_f)\,\Gamma^\mu(Q^2)\,u(p_i).")
    eq(r"\mathcal{M}=\frac{G_F\cos\theta_C}{\sqrt{2}}\,L_\mu J^\mu.")

    p("La cantidad relevante para la sección eficaz es el módulo cuadrado promediado:")
    eq(r"\overline{|\mathcal{M}|^2}=\frac{G_F^2\cos^2\theta_C}{2}\;\eta_{\mu\nu}\,W^{\mu\nu}.")
    eq(r"\eta_{\mu\nu}\equiv \sum_{\text{spins}} L_\mu(L_\nu)^\ast,\qquad W^{\mu\nu}\equiv \frac{1}{2}\sum_{\text{spins}} J^\mu(J^\nu)^\ast.")


def render_ch4() -> None:
    section("4. Tensor leptónico", r"Corriente $V\!-\!A$ y término antisimétrico ($\nu$ vs $\bar\nu$)")

    p("Corriente leptónica:")
    eq(r"L^\mu=\bar u(k')\,\gamma^\mu(1-\gamma^5)\,u(k).")

    p("Usando completitud, el tensor leptónico se expresa mediante una traza:")
    eq(r"\eta^{\mu\nu}=\mathrm{Tr}\!\left[(\not{k'}+m_\ell)\gamma^\mu(1-\gamma^5)\not{k}\gamma^\nu(1-\gamma^5)\right].")

    p("Resultado final (demostración en el Apéndice):")
    eq(r"""
\eta^{\mu\nu}
=
8\Big(
k^\mu k'^\nu+k^\nu k'^\mu
-g^{\mu\nu}(k\!\cdot\!k' - m_\ell^2)
\;\pm\;
i\,\epsilon^{\mu\nu\alpha\beta}k_\alpha k'_\beta
\Big).
""")
    st.info("La evaluación por trazas se encuentra en el **Apéndice**.")


def render_ch5() -> None:
    section("5. Corriente hadrónica", "Descomposición covariante y factores de forma")

    p("Definición:")
    eq(r"J^\mu=\bar u(p_f)\,\Gamma^\mu(Q^2)\,u(p_i),\qquad \Gamma^\mu=\Gamma_V^\mu-\Gamma_A^\mu.")

    p("Descomposición general compatible con Lorentz:")
    eq(r"""
\Gamma_V^\mu=
F_1^V(Q^2)\gamma^\mu
+i\frac{F_2^V(Q^2)}{2M}\sigma^{\mu\nu}Q_\nu
+\frac{F_3^V(Q^2)}{M}Q^\mu,
""")
    eq(r"""
\Gamma_A^\mu=
G_A(Q^2)\gamma^\mu\gamma^5
+\frac{G_P(Q^2)}{2M}Q^\mu\gamma^5
+i\frac{G_T(Q^2)}{2M}\sigma^{\mu\nu}Q_\nu\gamma^5.
""")

    p("Hipótesis estándar:")
    eq(r"\mathrm{CVC}\Rightarrow F_3^V=0,\qquad \text{corrientes de segunda clase}\Rightarrow G_T=0.")

    p("Bajo estas hipótesis, el vértice queda:")
    eq(r"""
\Gamma^\mu=
F_1^V\gamma^\mu
+i\frac{F_2^V}{2M}\sigma^{\mu\nu}Q_\nu
-\left(G_A\gamma^\mu\gamma^5+\frac{G_P}{2M}Q^\mu\gamma^5\right).
""")

    p(r"Conexión con factores de Sachs isovectoriales, usando $\tau=|Q^2|/(4M^2)$:")
    eq(r"F_1^V=\frac{G_E^V+\tau G_M^V}{1+\tau},\qquad F_2^V=\frac{G_M^V-G_E^V}{1+\tau}.")

    p("Sector axial (modelo dipolar, si se adopta):")
    eq(r"G_A(Q^2)=\frac{g_A}{\left(1+\frac{|Q^2|}{M_A^2}\right)^2},\qquad G_A(0)=g_A.")

    p("Relación PCAC / polo del pion (forma estándar):")
    eq(r"G_P(Q^2)\simeq \frac{4M^2}{m_\pi^2+|Q^2|}\,G_A(Q^2).")


def render_ch6() -> None:
    section("6. Tensor hadrónico", "Definición por traza y funciones estructura")

    p("Para blanco no polarizado:")
    eq(r"W^{\mu\nu}\equiv \frac{1}{2}\sum_{\text{spins}} J^\mu(J^\nu)^\ast.")

    p("Definimos el conjugado de Dirac del vértice:")
    eq(r"\bar{\Gamma}^{\nu}\equiv \gamma^0(\Gamma^\nu)^\dagger\gamma^0.")

    p("Usando completitud, se obtiene:")
    # Feynman slash: use \not{p} (KaTeX OK). Avoid \slashed which KaTeX doesn't support.
    eq(r"W^{\mu\nu}=\frac{1}{2}\,\mathrm{Tr}\!\left[(\not{p}_f+M)\Gamma^\mu(\not{p}_i+M)\bar{\Gamma}^\nu\right].")

    p("Descomposición covariante más general (funciones estructura):")
    eq(r"""
W^{\mu\nu}=
-g^{\mu\nu}W_1
+\frac{p_i^\mu p_i^\nu}{M^2}W_2
-i\frac{\epsilon^{\mu\nu\alpha\beta}p_{i\alpha}Q_\beta}{2M^2}W_3
+\frac{Q^\mu Q^\nu}{M^2}W_4
+\frac{p_i^\mu Q^\nu+Q^\mu p_i^\nu}{2M^2}W_5
+i\frac{p_i^\mu Q^\nu-Q^\mu p_i^\nu}{2M^2}W_6.
""")
    st.info(r"La evaluación explícita de trazas para obtener $W_i$ en términos de factores de forma se presenta en el **Apéndice**.")


def render_ch7() -> None:
    section("7. Contracción", r"Estructura de $\eta_{\mu\nu}W^{\mu\nu}$ y asimetría $\nu/\bar\nu$")

    p("El núcleo escalar es:")
    eq(r"\eta_{\mu\nu}W^{\mu\nu}.")

    p("Separación simétrica/antisimétrica:")
    eq(r"\eta_{\mu\nu}W^{\mu\nu}=\eta^{S}_{\mu\nu}W_S^{\mu\nu}+\eta^{A}_{\mu\nu}W_A^{\mu\nu}.")

    p("En la base de funciones estructura se escribe de forma modular:")
    eq(r"\eta_{\mu\nu}W^{\mu\nu}=\sum_{i=1}^{6}\mathcal{C}_i\,W_i.")

    # IMPORTANT: raw string so \bar{\nu} doesn't become backspace/newline escapes
    st.info(r"El término antisimétrico (típicamente asociado a $W_3$) controla la diferencia $\nu$ frente a $\bar{\nu}$.")


def render_ch8() -> None:
    section("8. Sección eficaz", r"Expresión final para $d\sigma/d|Q^2|$ y conexión angular")

    p("Para blanco en reposo y proceso $2\to2$:")
    eq(r"\frac{d\sigma}{dQ^2}=\frac{1}{64\pi\,M^2\,E_\nu^2}\;\overline{|\mathcal{M}|^2}.")

    p(r"Como $Q^2=-|Q^2|$:")
    eq(r"\frac{d\sigma}{d|Q^2|}=\frac{1}{64\pi\,M^2\,E_\nu^2}\;\overline{|\mathcal{M}|^2}.")

    p("Sustituyendo la forma tensorial:")
    eq(r"\frac{d\sigma}{d|Q^2|}=\frac{G_F^2\cos^2\theta_C}{128\pi\,M^2\,E_\nu^2}\;\eta_{\mu\nu}W^{\mu\nu}.")

    p("Forma modular en términos de funciones estructura:")
    eq(r"\frac{d\sigma}{d|Q^2|}=\frac{G_F^2\cos^2\theta_C}{128\pi\,M^2\,E_\nu^2}\;\sum_{i=1}^{6}\mathcal{C}_iW_i.")

    p("Cambio de variable angular (formal):")
    eq(r"\frac{d\sigma}{d\cos\theta_\ell}=\frac{d\sigma}{d|Q^2|}\left|\frac{d|Q^2|}{d\cos\theta_\ell}\right|.")

    p("Relación cinemática:")
    eq(r"|Q^2|=2E_\nu(E_\ell-p_\ell\cos\theta_\ell)-m_\ell^2,\qquad p_\ell=\sqrt{E_\ell^2-m_\ell^2}.")


def render_ch9() -> None:
    section("9. Aplicación numérica", "Esquema de evaluación para observables diferenciales")

    p(r"""
En aplicaciones numéricas resulta natural seguir un esquema modular basado en:
(i) cinemática, (ii) tensor leptónico, (iii) tensor hadrónico, y (iv) contracción tensorial.
""")

    p("Relación central:")
    eq(r"\overline{|\mathcal{M}|^2}=\frac{G_F^2\cos^2\theta_C}{2}\;\eta_{\mu\nu}W^{\mu\nu}.")
    p("y, en el laboratorio,")
    eq(r"\frac{d\sigma}{d|Q^2|}=\frac{G_F^2\cos^2\theta_C}{128\pi\,M^2\,E_\nu^2}\;\eta_{\mu\nu}W^{\mu\nu}.")

    st.info("Los detalles algebraicos extensos se recogen en el **Apéndice**.")


# ------------------------------------------------------------
# Routing
# ------------------------------------------------------------
Page = Tuple[str, Callable[[], None]]  # (label, renderer)

PAGES: Dict[str, Page] = {
    "home": ("Portada", render_home),
    "ch1": ("Notación", render_ch1),
    "ch2": ("Cinemática", render_ch2),
    "ch3": ("Amplitud CC", render_ch3),
    "ch4": ("Tensor leptónico", render_ch4),
    "ch5": ("Corriente hadrónica", render_ch5),
    "ch6": ("Tensor hadrónico", render_ch6),
    "ch7": ("Contracción", render_ch7),
    "ch8": ("Sección eficaz", render_ch8),
    "ch9": ("Aplicación numérica", render_ch9),
    "appendix": ("Apéndice", lambda: (section("Apéndice", "Trazas y contracciones"), render_appendix())),
}

ORDER = ["home", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8", "ch9", "appendix"]

if "page" not in st.session_state:
    st.session_state["page"] = "home"

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("Theoretical Framework")
labels = [PAGES[k][0] for k in ORDER]
current_idx = ORDER.index(st.session_state["page"])

choice = st.sidebar.radio(
    "Secciones",
    options=list(range(len(ORDER))),
    format_func=lambda i: labels[i],
    index=current_idx,
)
st.session_state["page"] = ORDER[choice]

st.sidebar.divider()
st.sidebar.caption("Accesos rápidos")
if st.sidebar.button("Ir al Apéndice", use_container_width=True):
    st.session_state["page"] = "appendix"
    st.rerun()
if st.sidebar.button("Volver a Portada", use_container_width=True):
    st.session_state["page"] = "home"
    st.rerun()

# ------------------------------------------------------------
# Main render
# ------------------------------------------------------------
page_key = st.session_state["page"]
PAGES[page_key][1]()  # render

st.divider()

# Prev/Next navigation
i = ORDER.index(page_key)
cols = st.columns([1, 1, 2])
with cols[0]:
    if st.button("⬅️ Anterior", use_container_width=True, disabled=(i == 0)):
        st.session_state["page"] = ORDER[i - 1]
        st.rerun()
with cols[1]:
    if st.button("Siguiente ➡️", use_container_width=True, disabled=(i == len(ORDER) - 1)):
        st.session_state["page"] = ORDER[i + 1]
        st.rerun()
with cols[2]:
    if page_key in {"ch4", "ch6", "ch7"}:
        st.info("Los desarrollos algebraicos extensos asociados a este capítulo se recogen en el **Apéndice**.")