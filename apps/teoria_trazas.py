# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 17:42:29 2026

@author: User
"""

# apps/teoria_trazas.py
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# -----------------------------
# Page config (MUST be first)
# -----------------------------
st.set_page_config(
    page_title="Apéndice — Trazas y contracciones",
    page_icon="📘",
    layout="wide",
)

# -----------------------------
# Ensure repo root in sys.path (helps in Cloud & imports if needed)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# -----------------------------
# Helpers: Anchored headings
# -----------------------------
def h2(text: str, anchor: str):
    st.markdown(f"<h2 id='{anchor}'>{text}</h2>", unsafe_allow_html=True)

def h3(text: str, anchor: str):
    st.markdown(f"<h3 id='{anchor}'>{text}</h3>", unsafe_allow_html=True)

# -----------------------------
# Sidebar navigation that works inside WordPress iframe
# -----------------------------
st.sidebar.header("Navegación")

def jump_to(anchor: str, label: str):
    if st.sidebar.button(label, use_container_width=True):
        # Use parent document if embedded; fallback to current document.
        components.html(
            f"""
            <script>
              const doc = (window.parent && window.parent.document) ? window.parent.document : document;
              const el = doc.getElementById("{anchor}");
              if (el) {{
                el.scrollIntoView({{behavior: "smooth", block: "start"}});
              }}
            </script>
            """,
            height=0,
        )

jump_to("clifford", "Álgebra de Clifford y convenciones")
jump_to("trazas", "Trazas básicas")
jump_to("quirales", "Proyectores quirales")
jump_to("leptonico", "Tensor leptónico")
jump_to("gamma5", "Simplificación usando γ5")
jump_to("abc", "Evaluación T(A), T(B), T(C)")
jump_to("levi", "Contracciones Levi-Civita")
jump_to("hadronico", "Tensor hadrónico (V/A/VA)")
jump_to("vectorial", "Vectorial: f11, f22, f12")
jump_to("f11", "Cálculo de f11")
jump_to("f22", "Cálculo de f22")

# -----------------------------
# Main content
# -----------------------------
st.title("📘 Apéndice: Cálculo de trazas y contracciones")
st.caption(r"""
En este apéndice se recopilan las identidades y pasos intermedios utilizados en el cálculo
del tensor leptónico, del tensor hadrónico y de la contracción
\(\tilde{\eta}_{\mu\nu}\tilde{W}^{\mu\nu}\).
""")

# ============================================================
h2("Álgebra de Clifford y convenciones", "clifford")
st.markdown(r"""
Adoptamos unidades naturales (\(\hbar=c=1\)) y la convención \(\varepsilon^{0123}=+1\).
Las matrices gamma satisfacen:
""")
st.latex(r"\{\gamma^\mu,\gamma^\nu\}=2g^{\mu\nu}, \qquad \gamma_5 = i\gamma^0\gamma^1\gamma^2\gamma^3.")

# ============================================================
h2("Trazas básicas", "trazas")
st.latex(r"\mathrm{Tr}[\gamma^\mu\gamma^\nu] = 4g^{\mu\nu}.")
st.latex(r"\mathrm{Tr}[\gamma^\mu\gamma^\nu\gamma^\rho\gamma^\sigma] = 4\left(g^{\mu\nu}g^{\rho\sigma}-g^{\mu\rho}g^{\nu\sigma}+g^{\mu\sigma}g^{\nu\rho}\right).")
st.latex(r"\mathrm{Tr}[\gamma^\mu\gamma^\nu\gamma^\rho\gamma^\sigma\gamma_5] = -4i\,\varepsilon^{\mu\nu\rho\sigma}.")

with st.expander("Traza de 6 matrices gamma (sin γ5)"):
    st.latex(r"""
\mathrm{Tr}\!\left[\gamma^{\mu_1}\gamma^{\mu_2}\gamma^{\mu_3}\gamma^{\mu_4}\gamma^{\mu_5}\gamma^{\mu_6}\right]
=4\,\Big(
 g^{\mu_1\mu_2}g^{\mu_3\mu_4}g^{\mu_5\mu_6}
- g^{\mu_1\mu_2}g^{\mu_3\mu_5}g^{\mu_4\mu_6}
+ g^{\mu_1\mu_2}g^{\mu_3\mu_6}g^{\mu_4\mu_5}
""")
    st.latex(r"""
- g^{\mu_1\mu_3}g^{\mu_2\mu_4}g^{\mu_5\mu_6}
+ g^{\mu_1\mu_3}g^{\mu_2\mu_5}g^{\mu_4\mu_6}
- g^{\mu_1\mu_3}g^{\mu_2\mu_6}g^{\mu_4\mu_5}
""")
    st.latex(r"""
+ g^{\mu_1\mu_4}g^{\mu_2\mu_3}g^{\mu_5\mu_6}
- g^{\mu_1\mu_4}g^{\mu_2\mu_5}g^{\mu_3\mu_6}
+ g^{\mu_1\mu_4}g^{\mu_2\mu_6}g^{\mu_3\mu_5}
""")
    st.latex(r"""
- g^{\mu_1\mu_5}g^{\mu_2\mu_3}g^{\mu_4\mu_6}
+ g^{\mu_1\mu_5}g^{\mu_2\mu_4}g^{\mu_3\mu_6}
- g^{\mu_1\mu_5}g^{\mu_2\mu_6}g^{\mu_3\mu_4}
""")
    st.latex(r"""
+ g^{\mu_1\mu_6}g^{\mu_2\mu_3}g^{\mu_4\mu_5}
- g^{\mu_1\mu_6}g^{\mu_2\mu_4}g^{\mu_3\mu_5}
+ g^{\mu_1\mu_6}g^{\mu_2\mu_5}g^{\mu_3\mu_4}
\Big).
""")

st.markdown("Además, la traza de un número impar de matrices gamma es nula.")

# ============================================================
h2("Proyectores quirales", "quirales")
st.markdown(r"Usaremos \(P_L=\frac{1-\gamma_5}{2}\), \(P_R=\frac{1+\gamma_5}{2}\), de modo que")
st.latex(r"(1-\gamma_5)^2 = 2(1-\gamma_5).")

# ============================================================
h2("Cálculo del tensor leptónico", "leptonico")

st.markdown("Introducimos la notación:")
st.latex(r"\tilde{\gamma}_\mu \equiv \gamma_\mu(1-\gamma_5) = \tilde{\gamma}_\mu^{V}+\tilde{\gamma}_\mu^{A}.")
st.latex(r"\not{k}\equiv \gamma_\alpha k^\alpha.")

st.markdown(r"""
Para el cálculo por trazas (aplicando completitud de espinores), aparece el objeto:
""")
st.latex(r"""
\mathcal{T}_{\mu \nu}
=
\mathrm{Tr}\!\left[
\frac{\not{k}_f+m_\mu}{2m_\mu}\,\tilde{\gamma}_\mu\,
\frac{\not{k}_i+m_\nu}{2m_\nu}\,\tilde{\gamma}_\nu
\right].
""")

st.markdown(r"""
Sustituyendo \(\tilde{\gamma}_\mu=\gamma_\mu(1-\gamma_5)\) y \(\tilde{\gamma}_\nu=\gamma_\nu(1-\gamma_5)\), y desarrollando:
""")

st.latex(r"""
\mathcal{T}_{\mu \nu}
=
\frac{1}{4m_\mu m_\nu}\,
\mathrm{Tr}\!\Big[
\not{k}_f\gamma_\mu(1-\gamma_5)\not{k}_i\gamma_\nu(1-\gamma_5)
+\not{k}_f\gamma_\mu(1-\gamma_5)\,m_\nu\gamma_\nu(1-\gamma_5)
+m_\mu\gamma_\mu(1-\gamma_5)\not{k}_i\gamma_\nu(1-\gamma_5)
+m_\mu\gamma_\mu(1-\gamma_5)\,m_\nu\gamma_\nu(1-\gamma_5)
\Big].
""")

st.markdown("A partir de aquí, separamos la contribución total en cuatro trazas:")
st.latex(r"""
\mathcal{T}_{\mu \nu}=\frac{1}{4m_\mu m_\nu}\,
\Big(T_{\mu \nu}^{(1)}+T_{\mu \nu}^{(2)}+T_{\mu \nu}^{(3)}+T_{\mu \nu}^{(4)}\Big).
""")

st.latex(r"""
T_{\mu \nu}^{(1)}
\equiv
\mathrm{Tr}\!\left[\not{k}_f\gamma_\mu(1-\gamma_5)\not{k}_i\gamma_\nu(1-\gamma_5)\right].
""")
st.latex(r"""
T_{\mu \nu}^{(2)}
\equiv
m_\nu\,\mathrm{Tr}\!\left[\not{k}_f\gamma_\mu(1-\gamma_5)\gamma_\nu(1-\gamma_5)\right].
""")
st.latex(r"""
T_{\mu \nu}^{(3)}
\equiv
m_\mu\,\mathrm{Tr}\!\left[\gamma_\mu(1-\gamma_5)\not{k}_i\gamma_\nu(1-\gamma_5)\right].
""")
st.latex(r"""
T_{\mu \nu}^{(4)}
\equiv
m_\mu m_\nu\,\mathrm{Tr}\!\left[\gamma_\mu(1-\gamma_5)\gamma_\nu(1-\gamma_5)\right].
""")

# ============================================================
h2("Simplificación usando γ5", "gamma5")
st.markdown("Usamos:")
st.latex(r"""
\{\gamma_5,\gamma^\alpha\}=0,\qquad
\gamma_5^2=\mathbb{I},\qquad
\gamma_5\,\not{a}\,\gamma_5=-\not{a},\qquad
\gamma_5\,\gamma^\alpha\,\gamma_5=-\gamma^\alpha.
""")

st.markdown(r"En particular:")
st.latex(r"""
P_L\,\gamma^\alpha\,P_L=0,\qquad P_L=\frac{1-\gamma_5}{2}.
""")

st.markdown("Consideremos el núcleo matricial:")
st.latex(r"\gamma_\mu(1-\gamma_5)\,\not{k}_i\,\gamma_\nu(1-\gamma_5).")

st.markdown("Al expandir:")
st.latex(r"""
\gamma_\mu(1-\gamma_5)\,\not{k}_i\,\gamma_\nu(1-\gamma_5)
=
\gamma_\mu\not{k}_i\gamma_\nu
-\gamma_\mu\gamma_5\not{k}_i\gamma_\nu
-\gamma_\mu\not{k}_i\gamma_\nu\gamma_5
+\gamma_\mu\gamma_5\not{k}_i\gamma_\nu\gamma_5.
""")

st.markdown("El último término se simplifica como:")
st.latex(r"""
\gamma_\mu\gamma_5\not{k}_i\gamma_\nu\gamma_5
=\gamma_\mu(\gamma_5\not{k}_i\gamma_5)(\gamma_5\gamma_\nu\gamma_5)
=\gamma_\mu(-\not{k}_i)(-\gamma_\nu)
=\gamma_\mu\not{k}_i\gamma_\nu.
""")

st.markdown("Por tanto, en cualquier traza donde aparezca este bloque:")
st.latex(r"""
\mathrm{Tr}[\cdots\,\gamma_\mu(1-\gamma_5)\not{k}_i\gamma_\nu(1-\gamma_5)\,\cdots]
=
2\,\mathrm{Tr}[\cdots\,\gamma_\mu\not{k}_i\gamma_\nu\,\cdots]
-\mathrm{Tr}[\cdots\,\gamma_\mu\gamma_5\not{k}_i\gamma_\nu\,\cdots]
-\mathrm{Tr}[\cdots\,\gamma_\mu\not{k}_i\gamma_\nu\gamma_5\,\cdots].
""")

# ============================================================
h2("Evaluación de T(A), T(B) y T(C)", "abc")
st.latex(r"T_{\mu\nu}=2T_{\mu\nu}^{(A)}-T_{\mu\nu}^{(B)}-T_{\mu\nu}^{(C)}.")

st.markdown("**(A) Término vectorial**")
st.latex(r"""
T_{\mu\nu}^{(A)} \equiv 
\mathrm{Tr}\!\left[\not{k}_f\,\gamma_\mu\,\not{k}_i\,\gamma_\nu\right]
=4\left(k_{f\mu}k_{i\nu}+k_{f\nu}k_{i\mu}-(k_f\!\cdot\!k_i)\,g_{\mu\nu}\right).
""")

st.markdown("**(B) Término axial (con \(\gamma_5\))**")
st.latex(r"""
T_{\mu\nu}^{(B)} \equiv 
\mathrm{Tr}\!\left[\not{k}_f\,\gamma_\mu\,\not{k}_i\,\gamma_\nu\,\gamma_5\right]
=-4i\,\varepsilon_{\alpha\mu\beta\nu}\,k_f^\alpha k_i^\beta.
""")

st.markdown("**(C) Segundo término axial (orden equivalente)**")
st.latex(r"""
T_{\mu\nu}^{(C)} \equiv 
\mathrm{Tr}\!\left[\not{k}_f\,\gamma_\mu\,\not{k}_i\,\gamma_\nu\,\gamma_5\right]
=-4i\,\varepsilon_{\alpha\mu\beta\nu}\,k_f^\alpha k_i^\beta,
\qquad T_{\mu\nu}^{(B)}=T_{\mu\nu}^{(C)}.
""")

st.markdown("**Resultado final**")
st.latex(r"""
T_{\mu\nu}
=
8\left(k_{f\mu}k_{i\nu}+k_{f\nu}k_{i\mu}-(k_f\!\cdot\!k_i)\,g_{\mu\nu}\right)
+8i\,\varepsilon_{\alpha\mu\beta\nu}\,k_f^\alpha k_i^\beta.
""")

# ============================================================
h2("Contracciones de tensores de Levi-Civita", "levi")
st.latex(r"""
\varepsilon_{\alpha\mu\beta\nu}\,\varepsilon^{\mu\nu\rho\lambda}
=-2\Big(\delta_\alpha^{\ \rho}\delta_\beta^{\ \lambda}-\delta_\alpha^{\ \lambda}\delta_\beta^{\ \rho}\Big).
""")
st.latex(r"""
A^\alpha B^\beta\,\varepsilon_{\alpha\mu\beta\nu}\,\varepsilon^{\mu\nu\rho\lambda}
=-2\left(A^\rho B^\lambda - A^\lambda B^\rho\right).
""")

# ============================================================
h2("Tensor hadrónico: descomposición vectorial, axial y vector–axial", "hadronico")
st.markdown("Partimos de la definición (promedio sobre el espín inicial y suma sobre el final):")
st.latex(r"""
\tilde{W}^{\mu \nu}
=
\frac{1}{8M^2}\,
\mathrm{Tr}\!\left[(\not{p}_f+M)\,\tilde{\Gamma}^{\mu}\,(\not{p}_i+M)\,\overline{\tilde{\Gamma}^{\nu}}\right].
""")

st.markdown("En el régimen cuasi-elástico, el vértice hadrónico efectivo se parametriza como:")
st.latex(r"""
\tilde{\Gamma}^{\mu}
=
F_1^V\,\gamma^{\mu}
+i\,\frac{F_2^V}{2M}\,\sigma^{\mu\alpha}Q_\alpha
+G_A\,\gamma^{\mu}\gamma^{5}
+F_P\,Q^{\mu}\gamma^{5},
\qquad
\sigma^{\mu\nu}\equiv \frac{i}{2}[\gamma^\mu,\gamma^\nu].
""")
st.latex(r"""
\overline{\tilde{\Gamma}^{\nu}}
=
F_1^V\,\gamma^{\nu}
-i\,\frac{F_2^V}{2M}\,\sigma^{\nu\beta}Q_\beta
+G_A\,\gamma^{\nu}\gamma^{5}
-F_P\,Q^{\nu}\gamma^{5}.
""")

st.markdown("Separación V/A:")
st.latex(r"""
\tilde{\Gamma}^{\mu}=\tilde{\Gamma}^{\mu}_{V}+\tilde{\Gamma}^{\mu}_{A},
\quad
\tilde{\Gamma}^{\mu}_{V}=F_1^V\,\gamma^{\mu}+i\,\frac{F_2^V}{2M}\,\sigma^{\mu\alpha}Q_\alpha,
\quad
\tilde{\Gamma}^{\mu}_{A}=G_A\,\gamma^{\mu}\gamma^{5}+F_P\,Q^{\mu}\gamma^{5}.
""")

st.markdown(r"Con ello \(\tilde{W}^{\mu\nu}=\tilde{W}^{\mu\nu}_V+\tilde{W}^{\mu\nu}_A+\tilde{W}^{\mu\nu}_{VA}\).")

# ============================================================
h2("Término vectorial", "vectorial")
st.latex(r"f_{11}^{\mu\nu},\; f_{22}^{\mu\nu}\; \text{y}\; f_{12}^{\mu\nu}")

st.latex(r"""
\tilde{W}^{\mu\nu}_{V}
=\frac{1}{8M^2}\,
\mathrm{Tr}\!\left[(\not{p}_f+M)\,\tilde{\Gamma}^{\mu}_{V}\,(\not{p}_i+M)\,\overline{\tilde{\Gamma}^{\nu}_{V}}\right].
""")

st.latex(r"""
\tilde{W}^{\mu\nu}_{V}
=
\frac{1}{8M^2}\left[
(F_1^V)^2\,f_{11}^{\mu\nu}
+\frac{(F_2^V)^2}{4M^2}\,f_{22}^{\mu\nu}
+\frac{i\,F_1^V F_2^V}{2M}\,f_{12}^{\mu\nu}
\right].
""")

st.markdown("donde definimos:")
st.latex(r"""
f_{11}^{\mu\nu}=
\mathrm{Tr}\!\left[(\not{p}_f+M)\,\gamma^{\mu}\,(\not{p}_i+M)\,\gamma^{\nu}\right],
""")
st.latex(r"""
f_{22}^{\mu\nu}=
\mathrm{Tr}\!\left[(\not{p}_f+M)\,\sigma^{\mu\alpha}Q_\alpha\,(\not{p}_i+M)\,\sigma^{\nu\beta}Q_\beta\right],
""")
st.latex(r"""
f_{12}^{\mu\nu}=
\mathrm{Tr}\!\left[
(\not{p}_f+M)\,\sigma^{\mu\alpha}Q_\alpha\,(\not{p}_i+M)\,\gamma^{\nu}
-(\not{p}_f+M)\,\gamma^{\mu}\,(\not{p}_i+M)\,\sigma^{\nu\beta}Q_\beta
\right].
""")

# ============================================================
h2("Cálculo de", "f11")
st.latex(r"f_{11}^{\mu\nu}")
st.latex(r"""
f_{11}^{\mu\nu}
=
4\left[p_f^{\mu}p_i^{\nu}+p_f^{\nu}p_i^{\mu}+\left(M^2-p_f\!\cdot\!p_i\right)g^{\mu\nu}\right].
""")

# ============================================================
h2("Cálculo de", "f22")
st.latex(r"f_{22}^{\mu\nu}")

st.markdown(r"Partimos de:")
st.latex(r"""
f_{22}^{\mu\nu}\equiv 
\mathrm{Tr}\!\left[(\not{p}_f+M)\,\sigma^{\mu\alpha}Q_\alpha\,(\not{p}_i+M)\,\sigma^{\nu\beta}Q_\beta\right].
""")

st.markdown("**1) Expansión en masas y cancelación de trazas impares**")
st.latex(r"""
f_{22}^{\mu\nu}
=
\mathrm{Tr}\!\left[\not{p}_f\,\sigma^{\mu\alpha}Q_\alpha\,\not{p}_i\,\sigma^{\nu\beta}Q_\beta\right]
+M^2\,\mathrm{Tr}\!\left[\sigma^{\mu\alpha}Q_\alpha\,\sigma^{\nu\beta}Q_\beta\right].
""")

st.markdown("**2) Término \(M^2\): traza \(\sigma\sigma\)**")
st.latex(r"""
\mathrm{Tr}\!\left(\sigma^{AB}\sigma^{CD}\right)=4\left(g^{AC}g^{BD}-g^{AD}g^{BC}\right),
\qquad
\mathrm{Tr}\!\left[\sigma^{\mu\alpha}Q_\alpha\,\sigma^{\nu\beta}Q_\beta\right]
=
4\left(g^{\mu\nu}Q^2-Q^\mu Q^\nu\right).
""")

st.markdown("**3) Término principal**")
st.latex(r"""
\sigma^{\mu\alpha}Q_\alpha=i\left(\gamma^\mu\not{Q}-Q^\mu\right),
\qquad
\sigma^{\nu\beta}Q_\beta=i\left(\gamma^\nu\not{Q}-Q^\nu\right).
""")

st.latex(r"""
\mathrm{Tr}\!\left[\not{p}_f\,\sigma^{\mu\alpha}Q_\alpha\,\not{p}_i\,\sigma^{\nu\beta}Q_\beta\right]
=
4\Big[
- Q^2\!\left(p_f^\mu p_i^\nu+p_f^\nu p_i^\mu\right)
+\left(p_f\!\cdot\!Q\right)\left(Q^\mu p_i^\nu+Q^\nu p_i^\mu\right)
+\left(p_i\!\cdot\!Q\right)\left(Q^\mu p_f^\nu+Q^\nu p_f^\mu\right)
-\left(p_f\!\cdot\!p_i\right)Q^\mu Q^\nu
+g^{\mu\nu}\Big(Q^2(p_f\!\cdot\!p_i)-2(p_f\!\cdot\!Q)(p_i\!\cdot\!Q)\Big)
\Big].
""")

st.markdown("**4) Resultado final para \(f_{22}^{\mu\nu}\)**")
st.latex(r"""
f_{22}^{\mu\nu}
=
4\Big[
- Q^2\!\left(p_f^\mu p_i^\nu+p_f^\nu p_i^\mu\right)
+\left(p_f\!\cdot\!Q\right)\left(Q^\mu p_i^\nu+Q^\nu p_i^\mu\right)
+\left(p_i\!\cdot\!Q\right)\left(Q^\mu p_f^\nu+Q^\nu p_f^\mu\right)
-\left(p_f\!\cdot\!p_i\right)Q^\mu Q^\nu
+g^{\mu\nu}\Big(Q^2(p_f\!\cdot\!p_i)-2(p_f\!\cdot\!Q)(p_i\!\cdot\!Q)\Big)
+M^2\left(g^{\mu\nu}Q^2-Q^\mu Q^\nu\right)
\Big].
""")

st.success(r"✅ El resultado es simétrico en \(\mu\leftrightarrow\nu\), como corresponde al término puramente vectorial.")