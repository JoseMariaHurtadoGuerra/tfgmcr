# apps/theoretical_framework/appendix.py
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components


def render_appendix() -> None:
    # IMPORTANT:
    # - Do NOT call st.set_page_config() here (only in app.py).
    # - Use st.latex for displayed equations (robust KaTeX rendering).

    st.markdown(
        r"""
<style>
.katex-display {
  overflow-x: auto;
  overflow-y: hidden;
  padding-bottom: 0.25rem;
}
</style>
""",
        unsafe_allow_html=True,
    )

    # -----------------------------
    # Anchored headings
    # -----------------------------
    def h2(text: str, anchor: str):
        st.markdown(f"<h2 id='{anchor}'>{text}</h2>", unsafe_allow_html=True)

    # -----------------------------
    # Sidebar navigation (appendix only)
    # -----------------------------
    st.sidebar.markdown("### Apéndice — Navegación interna")

    def jump_to(anchor: str, label: str):
        if st.sidebar.button(label, use_container_width=True):
            components.html(
                f"""
                <script>
                  (function() {{
                    function scrollIn(doc) {{
                      const el = doc.getElementById("{anchor}");
                      if (el) {{
                        el.scrollIntoView({{behavior: "smooth", block: "start"}});
                        return true;
                      }}
                      return false;
                    }}
                    const okHere = scrollIn(document);
                    if (!okHere && window.parent && window.parent.document) {{
                      scrollIn(window.parent.document);
                    }}
                  }})();
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
    jump_to("levi", "Contracciones Levi–Civita")
    jump_to("hadronico", "Tensor hadrónico (V/A/VA)")
    jump_to("vectorial", "Término vectorial")
    jump_to("f11", "Cálculo de f11")
    jump_to("f22", "Cálculo de f22")

    # -----------------------------
    # Main content
    # -----------------------------
    st.title("📘 Apéndice: Cálculo de trazas y contracciones")
    st.caption(
        r"""
En este apéndice se recopilan identidades y pasos intermedios utilizados en el cálculo
del tensor leptónico, del tensor hadrónico y de contracciones tensoriales.
"""
    )

    # ============================================================
    h2("Álgebra de Clifford y convenciones", "clifford")
    st.markdown(
        r"""
Se adoptan unidades naturales ($\hbar=c=1$) y la convención $\varepsilon^{0123}=+1$.
"""
    )
    st.latex(r"\{\gamma^\mu,\gamma^\nu\}=2g^{\mu\nu},\qquad \gamma^5=i\gamma^0\gamma^1\gamma^2\gamma^3.")
    st.latex(r"\not{a}\equiv \gamma_\mu a^\mu.")

    # ============================================================
    h2("Trazas básicas", "trazas")
    st.latex(r"\mathrm{Tr}[\gamma^\mu\gamma^\nu]=4g^{\mu\nu}.")
    st.latex(
        r"\mathrm{Tr}[\gamma^\mu\gamma^\nu\gamma^\rho\gamma^\sigma]=4\left(g^{\mu\nu}g^{\rho\sigma}-g^{\mu\rho}g^{\nu\sigma}+g^{\mu\sigma}g^{\nu\rho}\right)."
    )
    st.latex(r"\mathrm{Tr}[\gamma^\mu\gamma^\nu\gamma^\rho\gamma^\sigma\gamma^5]=-4i\,\varepsilon^{\mu\nu\rho\sigma}.")

    # ============================================================
    h2("Proyectores quirales", "quirales")
    st.markdown(r"Se emplean $P_L=\frac{1-\gamma^5}{2}$ y $P_R=\frac{1+\gamma^5}{2}$, de modo que")
    st.latex(r"(1-\gamma^5)^2=2(1-\gamma^5).")

    # ============================================================
    h2("Cálculo del tensor leptónico", "leptonico")
    st.markdown("Notación:")
    st.latex(r"\tilde{\gamma}_\mu\equiv \gamma_\mu(1-\gamma^5).")
    st.latex(r"\mathcal{T}_{\mu\nu}=\mathrm{Tr}\!\left[(\not{k}_f+m_\mu)\tilde{\gamma}_\mu(\not{k}_i+m_\nu)\tilde{\gamma}_\nu\right].")

    # ============================================================
    h2("Simplificación usando γ5", "gamma5")
    st.latex(r"\{\gamma^5,\gamma^\alpha\}=0,\qquad (\gamma^5)^2=\mathbb{I}.")
    st.latex(r"\gamma^5\not{a}\gamma^5=-\not{a},\qquad \gamma^5\gamma^\alpha\gamma^5=-\gamma^\alpha.")

    # ============================================================
    h2("Evaluación T(A), T(B), T(C)", "abc")
    st.latex(r"T_{\mu\nu}=2T_{\mu\nu}^{(A)}-T_{\mu\nu}^{(B)}-T_{\mu\nu}^{(C)}.")
    st.latex(
        r"T_{\mu\nu}^{(A)}=\mathrm{Tr}\!\left[\not{k}_f\gamma_\mu\not{k}_i\gamma_\nu\right]=4\left(k_{f\mu}k_{i\nu}+k_{f\nu}k_{i\mu}-(k_f\!\cdot\!k_i)g_{\mu\nu}\right)."
    )
    st.latex(
        r"T_{\mu\nu}^{(B)}=\mathrm{Tr}\!\left[\not{k}_f\gamma_\mu\not{k}_i\gamma_\nu\gamma^5\right]=-4i\,\varepsilon_{\alpha\mu\beta\nu}k_f^\alpha k_i^\beta."
    )

    # ============================================================
    h2("Contracciones Levi–Civita", "levi")
    st.latex(
        r"\varepsilon_{\alpha\mu\beta\nu}\varepsilon^{\mu\nu\rho\lambda}=-2\left(\delta_\alpha^{\ \rho}\delta_\beta^{\ \lambda}-\delta_\alpha^{\ \lambda}\delta_\beta^{\ \rho}\right)."
    )

    # ============================================================
    h2("Tensor hadrónico", "hadronico")
    st.markdown("Definición por traza (blanco no polarizado):")
    st.latex(
        r"\tilde{W}^{\mu\nu}=\frac{1}{8M^2}\mathrm{Tr}\!\left[(\not{p}_f+M)\tilde{\Gamma}^{\mu}(\not{p}_i+M)\overline{\tilde{\Gamma}^{\nu}}\right]."
    )

    # ============================================================
    h2("Término vectorial", "vectorial")
    st.latex(
        r"\tilde{W}^{\mu\nu}_{V}=\frac{1}{8M^2}\mathrm{Tr}\!\left[(\not{p}_f+M)\tilde{\Gamma}^{\mu}_{V}(\not{p}_i+M)\overline{\tilde{\Gamma}^{\nu}_{V}}\right]."
    )

    # ============================================================
    h2("Cálculo de f11", "f11")
    st.latex(
        r"f_{11}^{\mu\nu}=4\left[p_f^\mu p_i^\nu+p_f^\nu p_i^\mu+\left(M^2-p_f\!\cdot\!p_i\right)g^{\mu\nu}\right]."
    )

    # ============================================================
    h2("Cálculo de f22", "f22")
    st.latex(
        r"f_{22}^{\mu\nu}\equiv \mathrm{Tr}\!\left[(\not{p}_f+M)\sigma^{\mu\alpha}Q_\alpha(\not{p}_i+M)\sigma^{\nu\beta}Q_\beta\right]."
    )
    st.success("✅ Apéndice cargado correctamente.")