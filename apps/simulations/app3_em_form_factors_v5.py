from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Factores de forma EM e isovectoriales", layout="wide")

st.markdown(
    """
    <style>
    .katex-display {
        overflow-x: auto;
        overflow-y: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

M_N = 0.93891897
MU_P = 2.793
MU_N = -1.913
MU_V = MU_P - MU_N
MV_STD = 0.843
LAMBDA_N_STD = 5.6


def tau_qe(Q2: np.ndarray | float, M_N_val: float = M_N) -> np.ndarray:
    q2 = np.asarray(Q2, dtype=float)
    return q2 / (4.0 * M_N_val**2)


def dipole_gd(Q2: np.ndarray | float, M_V: float) -> np.ndarray:
    q2 = np.asarray(Q2, dtype=float)
    return 1.0 / (1.0 + q2 / M_V**2) ** 2


def lambda_d_from_mv(M_V: float, M_N_val: float = M_N) -> float:
    return 4.0 * M_N_val**2 / M_V**2


def galster_sachs(
    Q2: np.ndarray | float,
    M_V: float = MV_STD,
    lambda_n: float = LAMBDA_N_STD,
    M_N_val: float = M_N,
    mu_p: float = MU_P,
    mu_n: float = MU_N,
) -> dict[str, np.ndarray]:
    q2 = np.asarray(Q2, dtype=float)
    tau = tau_qe(q2, M_N_val)
    gd = dipole_gd(q2, M_V)
    xi_n = 1.0 / (1.0 + lambda_n * tau)

    gep = gd
    gen = -mu_n * tau * gd * xi_n
    gmp = mu_p * gd
    gmn = mu_n * gd

    return {"GEp": gep, "GEn": gen, "GMp": gmp, "GMn": gmn}


def _coerce_gkex_output(res: Any) -> dict[str, np.ndarray]:
    if isinstance(res, dict):
        aliases = {
            "GEp": ["GEp", "G_Ep", "gep", "GE_p"],
            "GEn": ["GEn", "G_En", "gen", "GE_n"],
            "GMp": ["GMp", "G_Mp", "gmp", "GM_p"],
            "GMn": ["GMn", "G_Mn", "gmn", "GM_n"],
        }
        out: dict[str, np.ndarray] = {}
        for target, keys in aliases.items():
            for key in keys:
                if key in res:
                    out[target] = np.asarray(res[key], dtype=float)
                    break
        if len(out) == 4:
            return out

    if isinstance(res, (list, tuple)) and len(res) == 4:
        return {
            "GEp": np.asarray(res[0], dtype=float),
            "GMp": np.asarray(res[1], dtype=float),
            "GEn": np.asarray(res[2], dtype=float),
            "GMn": np.asarray(res[3], dtype=float),
        }

    arr = np.asarray(res, dtype=float)
    if arr.ndim == 2 and 4 in arr.shape:
        if arr.shape[0] == 4:
            return {"GEp": arr[0], "GMp": arr[1], "GEn": arr[2], "GMn": arr[3]}
        if arr.shape[1] == 4:
            return {"GEp": arr[:, 0], "GMp": arr[:, 1], "GEn": arr[:, 2], "GMn": arr[:, 3]}

    raise ValueError("No se pudo interpretar la salida de sachs_gkex.")


def gkex_sachs(Q2: np.ndarray | float) -> dict[str, np.ndarray] | None:
    try:
        from src.form_factors_gkex import sachs_gkex  # type: ignore
    except Exception:
        return None

    q2 = np.asarray(Q2, dtype=float)
    sig = inspect.signature(sachs_gkex)
    params = sig.parameters

    def _build_kwargs(q2_value: Any) -> dict[str, Any]:
        if "Q2" in params:
            return {"Q2": q2_value}
        if "q2" in params:
            return {"q2": q2_value}
        if "Q2_GeV2" in params:
            return {"Q2_GeV2": q2_value}
        return {}

    try:
        return _coerce_gkex_output(sachs_gkex(**_build_kwargs(q2 if q2.ndim == 0 else q2)))
    except Exception:
        pass

    q2_flat = np.atleast_1d(q2).astype(float)
    values = []
    for q2i in q2_flat:
        values.append(_coerce_gkex_output(sachs_gkex(**_build_kwargs(float(q2i)))))

    stacked = {
        key: np.array([item[key] for item in values], dtype=float).reshape(q2_flat.shape)
        for key in ["GEp", "GMp", "GEn", "GMn"]
    }
    if np.asarray(Q2).ndim == 0:
        return {k: v[0] for k, v in stacked.items()}
    return stacked


EXPECTED_COLUMNS = {"panel", "Q2", "y"}
PANEL_ORDER = ["GEp", "GMp", "GEn", "GMn"]
PANEL_LABELS = {
    "GEp": r"$G_E^p/G_D^{\rm ref}$",
    "GMp": r"$G_M^p/(\mu_p G_D^{\rm ref})$",
    "GEn": r"$G_E^n/G_D^{\rm ref}$",
    "GMn": r"$G_M^n/(\mu_n G_D^{\rm ref})$",
}


def load_points_dataframe(uploaded_file: Any | None = None) -> pd.DataFrame | None:
    candidates = []
    if uploaded_file is not None:
        candidates.append(uploaded_file)
    repo_candidates = [
        Path("refs/em_form_factors/ff_points_green.csv"),
        Path("refs/em_form_factors/data_points.csv"),
        Path("data/processed/em_form_factors/ff_points_green.csv"),
    ]
    candidates.extend([p for p in repo_candidates if p.exists()])

    for candidate in candidates:
        try:
            df = pd.read_csv(candidate)
        except Exception:
            continue
        if not EXPECTED_COLUMNS.issubset(set(df.columns)):
            continue
        df = df.copy()
        if "yerr" not in df.columns:
            df["yerr"] = np.nan
        df["panel"] = df["panel"].astype(str)
        df = df[df["panel"].isin(PANEL_ORDER)].sort_values(["panel", "Q2"])
        return df
    return None


def safe_divide(num: np.ndarray, den: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.abs(den) > eps
    out[mask] = num[mask] / den[mask]
    return out


def ratios_from_sachs(sachs: dict[str, np.ndarray], gd_ref: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "GEp": safe_divide(sachs["GEp"], gd_ref),
        "GMp": safe_divide(sachs["GMp"], MU_P * gd_ref),
        "GEn": safe_divide(sachs["GEn"], gd_ref),
        "GMn": safe_divide(sachs["GMn"], MU_N * gd_ref),
    }


def isovector_from_sachs(sachs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "GEV": np.asarray(sachs["GEp"], dtype=float) - np.asarray(sachs["GEn"], dtype=float),
        "GMV": np.asarray(sachs["GMp"], dtype=float) - np.asarray(sachs["GMn"], dtype=float),
    }


def isovector_ratios_from_sachs(sachs: dict[str, np.ndarray], gd_ref: np.ndarray) -> dict[str, np.ndarray]:
    iso = isovector_from_sachs(sachs)
    return {
        "GEV": safe_divide(iso["GEV"], gd_ref),
        "GMV": safe_divide(iso["GMV"], MU_V * gd_ref),
    }


st.sidebar.header("Controles")
model_choice = st.sidebar.radio("Curvas a mostrar", ["Galster", "GKeX", "Ambas"], index=2)
q2_max = st.sidebar.slider(r"$|Q^2|_{\max}$ (GeV$^2$)", 0.10, 10.00, 10.00, 0.10)
n_points = st.sidebar.slider("Número de puntos de muestreo", 150, 1200, 500, 50)

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros de Galster")
M_V = st.sidebar.slider(r"$M_V$ (GeV)", 0.700, 1.000, MV_STD, 0.005)
lambda_n = st.sidebar.slider(r"$\lambda_n$", 3.0, 8.0, LAMBDA_N_STD, 0.1)
lambda_d = lambda_d_from_mv(M_V)
st.sidebar.latex(rf"\lambda_D^V = \frac{{4M_N^2}}{{M_V^2}} = {lambda_d:.3f}")
st.sidebar.markdown(r"$M_V$ y $\lambda_D^V$ quedan ligados automáticamente por definición.")

st.sidebar.markdown("---")
st.sidebar.subheader("Normalización del dipolo de referencia")
reference_mode = st.sidebar.radio(
    "Denominador en los cocientes",
    [
        "Normalización publicada (M_V^ref = 0.843 GeV)",
        "Normalización dinámica (M_V^ref = M_V actual)",
    ],
    index=0,
)

show_data = st.sidebar.checkbox("Mostrar puntos experimentales si hay CSV", value=True)
uploaded_csv = st.sidebar.file_uploader(
    "Sube CSV opcional con puntos verdes",
    type=["csv"],
    help="Columnas mínimas: panel,Q2,y y opcionalmente yerr. panel debe ser GEp, GMp, GEn o GMn.",
)
show_logx = st.sidebar.checkbox("Eje x logarítmico", value=True)
show_summary_table = st.sidebar.checkbox("Mostrar tabla de valores destacados", value=True)

st.title("Factores de forma del nucleón: EM e isovectoriales")
st.markdown(
    "Esta app reúne dos niveles de análisis. En la primera pestaña se comparan los factores de forma electromagnéticos de "
    "Sachs del protón y del neutrón. En la segunda se construyen las combinaciones isovectoriales que entran en la corriente "
    "débil vectorial mediante CVC. Así se ve cómo una misma elección fenomenológica en el sector electromagnético se "
    "propaga después al sector vectorial usado en procesos cuasielásticos."
)

Q2 = np.geomspace(1.0e-3, q2_max, n_points) if show_logx else np.linspace(1.0e-3, q2_max, n_points)
M_V_ref = MV_STD if reference_mode.startswith("Normalización publicada") else M_V
gd_ref = dipole_gd(Q2, M_V_ref)

galster_raw = galster_sachs(Q2, M_V=M_V, lambda_n=lambda_n)
galster_ratio = ratios_from_sachs(galster_raw, gd_ref)
galster_isovector_ratio = isovector_ratios_from_sachs(galster_raw, gd_ref)

gkex_ratio = None
gkex_isovector_ratio = None
gkex_error = None
if model_choice in {"GKeX", "Ambas"}:
    try:
        gkex_raw = gkex_sachs(Q2)
        if gkex_raw is not None:
            gkex_ratio = ratios_from_sachs(gkex_raw, gd_ref)
            gkex_isovector_ratio = isovector_ratios_from_sachs(gkex_raw, gd_ref)
        else:
            gkex_error = (
                "No se encontró el módulo src/form_factors_gkex.py en el proyecto actual. "
                "La parte Galster sigue funcionando y la comparación con GKeX se activará en cuanto ese archivo esté disponible."
            )
    except Exception as exc:
        gkex_error = str(exc)

points_df = load_points_dataframe(uploaded_csv) if show_data else None

tab_em, tab_iso = st.tabs(["Factores de forma EM", "Factores de forma isovectoriales"])

with tab_em:
    with st.expander("Marco teórico y lectura física", expanded=True):
        st.markdown("**Galster** es una parametrización sencilla y transparente para explorar sensibilidad paramétrica a bajo momento transferido.")
        st.latex(r"\tau = \frac{|Q^2|}{4M_N^2}")
        st.latex(r"G_D^V(Q^2) = \frac{1}{\left(1+\frac{|Q^2|}{M_V^2}\right)^2} = \frac{1}{(1+\lambda_D^V\tau)^2}")
        st.latex(r"G_E^p = G_D^V, \qquad G_E^n = -\mu_n\,\tau\,G_D^V\,\xi_n, \qquad \xi_n = \frac{1}{1+\lambda_n\tau}")
        st.latex(r"G_M^p = \mu_p G_D^V, \qquad G_M^n = \mu_n G_D^V")
        st.markdown("**GKeX** se utiliza como referencia fenomenológica más rica, manteniendo Galster como modelo editable.")
        st.markdown(r"""
**Qué se varía aquí y por qué**

- **$M_V$** fija la escala del dipolo: al aumentar $M_V$, la caída con $|Q^2|$ se hace más lenta.
- **$\lambda_n$** controla sobre todo la forma de $G_E^n$.
- **$|Q^2|_{\max}$** puede llevarse hasta $10\,\mathrm{GeV}^2$ para comparar mejor con las figuras de la tesis y ver cuándo GKeX empieza a separarse claramente del ansatz dipolar.
""")
        st.markdown(r"""
**Advertencia de normalización**

Los cocientes pueden mostrarse respecto al dipolo publicado con $M_V^{\rm ref}=0.843\,\mathrm{GeV}$ o respecto a un dipolo dinámico actualizado con el valor del slider.
""")

    fig_em, axes_em = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)
    axes_map_em = {"GEp": axes_em[0, 0], "GMp": axes_em[0, 1], "GEn": axes_em[1, 0], "GMn": axes_em[1, 1]}
    y_limits_em = {"GEp": (0.0, 1.2), "GMp": (0.90, 1.12), "GEn": (-0.05, 0.80), "GMn": (0.60, 1.15)}

    for panel in PANEL_ORDER:
        ax = axes_map_em[panel]
        if model_choice in {"Galster", "Ambas"}:
            ax.plot(Q2, galster_ratio[panel], label="Galster", linewidth=2.0)
        if model_choice in {"GKeX", "Ambas"} and gkex_ratio is not None:
            ax.plot(Q2, gkex_ratio[panel], label="GKeX", linewidth=2.0)
        if points_df is not None:
            sub = points_df[points_df["panel"] == panel]
            if len(sub) > 0:
                if sub["yerr"].notna().any():
                    ax.errorbar(sub["Q2"], sub["y"], yerr=sub["yerr"], fmt=".", capsize=2, linestyle="none", label="Datos" if panel == "GEp" else None)
                else:
                    ax.plot(sub["Q2"], sub["y"], ".", label="Datos" if panel == "GEp" else None)
        ax.set_ylabel(PANEL_LABELS[panel])
        ax.set_xlabel(r"$|Q^2|\;({\rm GeV}^2)$")
        ax.set_ylim(*y_limits_em[panel])
        if show_logx:
            ax.set_xscale("log")
        ax.grid(alpha=0.25)

    handles, _labels = axes_em[0, 0].get_legend_handles_labels()
    if handles:
        axes_em[0, 0].legend(frameon=False)
    st.pyplot(fig_em, use_container_width=True)
    if gkex_error is not None:
        st.warning(gkex_error)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Lectura física rápida")
        st.markdown(r"""
- Si aumentas **$M_V$**, el dipolo cae más lentamente y la curva de Galster se endurece.
- Si aumentas **$\lambda_n$**, el factor $\xi_n=(1+\lambda_n\tau)^{-1}$ suprime más deprisa a $G_E^n$.
- La comparación con **GKeX** muestra hasta qué punto un modelo VMD más rico se separa del ansatz dipolar simple.
""")
    with col2:
        st.subheader("Propósito didáctico")
        st.markdown(r"""
Esta visualización ayuda a distinguir tres ideas:
1. qué fija la normalización en $Q^2\to 0$,
2. qué controla la forma funcional de Galster, y
3. por qué GKeX actúa como referencia fenomenológica más completa en el sector vectorial.
""")

    if show_summary_table:
        st.subheader("Valores destacados")
        q2_marks = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0])
        q2_marks = q2_marks[q2_marks <= q2_max + 1e-12]
        gal_mark = galster_sachs(q2_marks, M_V=M_V, lambda_n=lambda_n)
        gal_ratio_mark = ratios_from_sachs(gal_mark, dipole_gd(q2_marks, M_V_ref))
        gk_ratio_mark = None
        if gkex_ratio is not None:
            gk = gkex_sachs(q2_marks)
            if gk is not None:
                gk_ratio_mark = ratios_from_sachs(gk, dipole_gd(q2_marks, M_V_ref))
        rows = []
        for i, qv in enumerate(q2_marks):
            row = {"Q2 [GeV^2]": float(qv)}
            for panel in PANEL_ORDER:
                row[f"Galster {panel}"] = float(gal_ratio_mark[panel][i])
            if gk_ratio_mark is not None:
                for panel in PANEL_ORDER:
                    row[f"GKeX {panel}"] = float(gk_ratio_mark[panel][i])
            rows.append(row)
        st.dataframe(pd.DataFrame.from_records(rows), use_container_width=True)

with tab_iso:
    with st.expander("Marco teórico y lectura física", expanded=True):
        st.markdown(
            "En esta sección se construyen las combinaciones isovectoriales asociadas al **sector vectorial de la corriente débil**. "
            "Bajo CVC, la parte vectorial relevante se obtiene a partir de las combinaciones protón menos neutrón."
        )
        st.latex(r"G_E^V(Q^2) = G_E^p(Q^2) - G_E^n(Q^2)")
        st.latex(r"G_M^V(Q^2) = G_M^p(Q^2) - G_M^n(Q^2), \qquad \mu_V = \mu_p - \mu_n")
        st.latex(r"\frac{G_E^V}{G_D^{\rm ref}}, \qquad \frac{G_M^V}{\mu_V G_D^{\rm ref}}")
        st.markdown(r"""
**Qué parámetros merece la pena variar aquí**

- No aparecen nuevos parámetros independientes. Esta pestaña hereda directamente la física del sector EM.
- Por eso tiene sentido seguir variando **$M_V$** y **$\lambda_n$**: ambos modifican $G_E^p$, $G_E^n$, $G_M^p$ y $G_M^n$, y por tanto deforman también $G_E^V$ y $G_M^V$.
- La cantidad **$\mu_V$** no conviene convertirla en slider: fija simplemente la normalización magnética isovectorial en $Q^2=0$.
- Si más adelante quieres añadir una pestaña axial aparte, entonces sí aparece un parámetro natural nuevo: **$M_A$**.
""")

    fig_iso, axes_iso = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    iso_labels = {"GEV": r"$G_E^V/G_D^{\rm ref}$", "GMV": r"$G_M^V/(\mu_V G_D^{\rm ref})$"}
    iso_limits = {"GEV": (-0.50, 1.10), "GMV": (0.85, 1.10)}

    for panel, ax in zip(["GEV", "GMV"], axes_iso):
        if model_choice in {"Galster", "Ambas"}:
            ax.plot(Q2, galster_isovector_ratio[panel], label="Galster", linewidth=2.0)
        if model_choice in {"GKeX", "Ambas"} and gkex_isovector_ratio is not None:
            ax.plot(Q2, gkex_isovector_ratio[panel], label="GKeX", linewidth=2.0)
        ax.set_ylabel(iso_labels[panel])
        ax.set_xlabel(r"$|Q^2|\;({\rm GeV}^2)$")
        ax.set_ylim(*iso_limits[panel])
        if show_logx:
            ax.set_xscale("log")
        ax.grid(alpha=0.25)

    handles, _labels = axes_iso[0].get_legend_handles_labels()
    if handles:
        axes_iso[0].legend(frameon=False)
    st.pyplot(fig_iso, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Lectura física rápida")
        st.markdown(r"""
- **$G_E^V$** mide directamente el balance entre el sector protónico y el neutrónico en la combinación isovectorial.
- En **Galster**, $G_M^V/(\mu_V G_D)$ queda exactamente plano en 1 porque tanto $G_M^p$ como $G_M^n$ son dipolares puros.
- En **GKeX**, la desviación respecto a 1 mide hasta qué punto el sector magnético deja de comportarse como un dipolo simple.
""")
    with col4:
        st.subheader("Por qué esta pestaña merece la pena")
        st.markdown(
            "Aquí se ve de forma inmediata qué parte de la estructura hadrónica introducida en el sector electromagnético se "
            "transmite después al vector débil. Es justamente la combinación útil para conectar las curvas de Sachs con la "
            "corriente vectorial que usarás luego en procesos cuasielásticos."
        )

    if show_summary_table:
        st.subheader("Valores destacados")
        q2_marks = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0])
        q2_marks = q2_marks[q2_marks <= q2_max + 1e-12]
        gal_mark = galster_sachs(q2_marks, M_V=M_V, lambda_n=lambda_n)
        gal_iso_mark = isovector_ratios_from_sachs(gal_mark, dipole_gd(q2_marks, M_V_ref))
        gk_iso_mark = None
        if gkex_isovector_ratio is not None:
            gk = gkex_sachs(q2_marks)
            if gk is not None:
                gk_iso_mark = isovector_ratios_from_sachs(gk, dipole_gd(q2_marks, M_V_ref))
        rows_iso = []
        for i, qv in enumerate(q2_marks):
            row = {
                "Q2 [GeV^2]": float(qv),
                "Galster GEV": float(gal_iso_mark["GEV"][i]),
                "Galster GMV": float(gal_iso_mark["GMV"][i]),
            }
            if gk_iso_mark is not None:
                row["GKeX GEV"] = float(gk_iso_mark["GEV"][i])
                row["GKeX GMV"] = float(gk_iso_mark["GMV"][i])
            rows_iso.append(row)
        st.dataframe(pd.DataFrame.from_records(rows_iso), use_container_width=True)
