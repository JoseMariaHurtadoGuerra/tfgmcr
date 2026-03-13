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


st.set_page_config(page_title="EM form factors: Galster vs GKeX", layout="wide")

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

# -----------------------------------------------------------------------------
# Physical constants (natural units)
# -----------------------------------------------------------------------------
M_N = 0.93891897  # GeV, average nucleon mass for tau definition
MU_P = 2.793
MU_N = -1.913
MV_STD = 0.843  # GeV, standard dipole mass used in Guillermo/Megias figures
LAMBDA_N_STD = 5.6


# -----------------------------------------------------------------------------
# Galster model
# -----------------------------------------------------------------------------
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

    return {
        "GEp": gep,
        "GEn": gen,
        "GMp": gmp,
        "GMn": gmn,
    }


# -----------------------------------------------------------------------------
# GKeX wrapper
# -----------------------------------------------------------------------------
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
        # Tu modulo devuelve (GEp, GMp, GEn, GMn)
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

    raise ValueError(
        "No se pudo interpretar la salida de sachs_gkex. "
        "Adapta la función _coerce_gkex_output a la firma real de tu módulo src/form_factors_gkex.py."
    )



def gkex_sachs(Q2: np.ndarray | float) -> dict[str, np.ndarray] | None:
    try:
        from src.form_factors_gkex import sachs_gkex  # type: ignore
    except Exception:
        return None

    q2 = np.asarray(Q2, dtype=float)
    sig = inspect.signature(sachs_gkex)
    params = sig.parameters

    def _build_kwargs(q2_value: float) -> dict[str, float]:
        if "Q2" in params:
            return {"Q2": float(q2_value)}
        if "q2" in params:
            return {"q2": float(q2_value)}
        if "Q2_GeV2" in params:
            return {"Q2_GeV2": float(q2_value)}
        return {}

    # Si el backend acepta arrays, perfecto. Si no, caemos automaticamente a evaluacion punto a punto.
    try:
        kwargs = _build_kwargs(q2 if q2.ndim == 0 else q2)
        res = sachs_gkex(**kwargs)
        return _coerce_gkex_output(res)
    except Exception:
        pass

    q2_flat = np.atleast_1d(q2).astype(float)
    values = []
    last_exc = None
    for q2i in q2_flat:
        try:
            values.append(_coerce_gkex_output(sachs_gkex(**_build_kwargs(float(q2i)))))
        except Exception as exc:
            last_exc = exc
            raise RuntimeError(
                "Se encontro src.form_factors_gkex.sachs_gkex, pero el backend no pudo evaluarse ni en modo vectorial ni punto a punto. "
                f"Ultimo error: {last_exc}"
            ) from exc

    stacked = {
        key: np.array([item[key] for item in values], dtype=float).reshape(q2_flat.shape)
        for key in ["GEp", "GMp", "GEn", "GMn"]
    }

    if np.asarray(Q2).ndim == 0:
        return {k: v[0] for k, v in stacked.items()}
    return stacked


# -----------------------------------------------------------------------------
# Experimental points loader
# -----------------------------------------------------------------------------
EXPECTED_COLUMNS = {"panel", "Q2", "y"}
OPTIONAL_COLUMNS = {"yerr"}
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

        cols = set(df.columns)
        if not EXPECTED_COLUMNS.issubset(cols):
            continue
        df = df.copy()
        if "yerr" not in df.columns:
            df["yerr"] = np.nan
        df["panel"] = df["panel"].astype(str)
        df = df[df["panel"].isin(PANEL_ORDER)].sort_values(["panel", "Q2"])
        return df
    return None


# -----------------------------------------------------------------------------
# Ratios shown in the figure
# -----------------------------------------------------------------------------
def ratios_from_sachs(sachs: dict[str, np.ndarray], gd_ref: np.ndarray) -> dict[str, np.ndarray]:
    eps = 1e-15

    def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        num = np.asarray(num, dtype=float)
        den = np.asarray(den, dtype=float)
        out = np.full_like(num, np.nan, dtype=float)
        mask = np.abs(den) > eps
        out[mask] = num[mask] / den[mask]
        return out

    return {
        "GEp": safe_div(sachs["GEp"], gd_ref),
        "GMp": safe_div(sachs["GMp"], MU_P * gd_ref),
        "GEn": safe_div(sachs["GEn"], gd_ref),
        "GMn": safe_div(sachs["GMn"], MU_N * gd_ref),
    }


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Controles")
model_choice = st.sidebar.radio(
    "Curvas a mostrar",
    options=["Galster", "GKeX", "Ambas"],
    index=2,
)

q2_max = st.sidebar.slider(
    r"$|Q^2|_{\max}$ (GeV$^2$)",
    min_value=0.10,
    max_value=1.00,
    value=1.00,
    step=0.05,
)

n_points = st.sidebar.slider("Número de puntos de muestreo", 150, 1200, 500, 50)

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros de Galster")
M_V = st.sidebar.slider(r"$M_V$ (GeV)", 0.700, 1.000, MV_STD, 0.005)
lambda_n = st.sidebar.slider(r"$\lambda_n$", 3.0, 8.0, LAMBDA_N_STD, 0.1)
lambda_d = lambda_d_from_mv(M_V)
st.sidebar.latex(rf"\lambda_D^V = \frac{{4M_N^2}}{{M_V^2}} = {lambda_d:.3f}")
st.sidebar.caption(r"Aquí $M_V$ y $\lambda_D^V$ están ligados físicamente por definición.")

st.sidebar.markdown("---")
st.sidebar.subheader("Normalización del dipolo de referencia")
reference_mode = st.sidebar.radio(
    "Denominador en los cocientes",
    options=[
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


# -----------------------------------------------------------------------------
# Page header and theory block
# -----------------------------------------------------------------------------
st.title("Factores de forma electromagnéticos del nucleón: Galster vs GKeX")

st.markdown(
    "Esta app reproduce de forma interactiva la comparación entre las parametrizaciones "
    "Galster y GKeX para los factores de forma de Sachs del nucleón, siguiendo el espíritu de la "
    "figura mostrada en la tesis. El objetivo es separar claramente qué parte de la física pertenece "
    "a un ansatz simple y manipulable (Galster) y qué parte pertenece a un modelo fenomenológico "
    "más sofisticado (GKeX)."
)

with st.expander("Marco teórico y lectura física", expanded=True):
    st.markdown("**Galster.** Es una parametrización sencilla, útil para explorar sensibilidad paramétrica a bajo momento transferido.")
    st.latex(r"\tau = \frac{|Q^2|}{4M_N^2}")
    st.latex(r"G_D^V(Q^2) = \frac{1}{\left(1+\frac{|Q^2|}{M_V^2}\right)^2} = \frac{1}{(1+\lambda_D^V\tau)^2}")
    st.latex(r"G_E^p = G_D^V, \qquad G_E^n = -\mu_n\,\tau\,G_D^V\,\xi_n, \qquad \xi_n = \frac{1}{1+\lambda_n\tau}")
    st.latex(r"G_M^p = \mu_p G_D^V, \qquad G_M^n = \mu_n G_D^V")

    st.markdown("**GKeX.** Es una extensión del enfoque de dominancia vector-mesón (VMD), conectada con la fenomenología de mayor |Q²|. En esta primera versión se usa como curva de referencia sin sliders internos.")
    st.caption("Backend esperado: `sachs_gkex(Q2, M=0.939565, p=GKex05Params()) -> (GEp, GMp, GEn, GMn)`.")

    st.markdown("**Qué se varía aquí y por qué.**")
    st.markdown(
        r"- **$M_V$** controla la escala del dipolo: al aumentar $M_V$, la caída con $|Q^2|$ se hace más lenta.  \n"
        r"- **$\lambda_n$** controla específicamente la forma de $G_E^n$.  \n"
        r"- **$|Q^2|_{\max}$** se restringe a $\leq 1\,\mathrm{GeV}^2$ para mantener la exploración centrada en el rango donde Galster es didácticamente más razonable."
    )

    st.markdown(r"**Advertencia importante.** En la literatura, los cocientes suelen mostrarse respecto a un dipolo de referencia con $M_V^{\rm ref}=0.843\,$GeV. Por eso esta app permite distinguir entre la normalización publicada y una normalización dinámica.")


# -----------------------------------------------------------------------------
# Curves
# -----------------------------------------------------------------------------
Q2 = np.geomspace(1.0e-3, q2_max, n_points) if show_logx else np.linspace(1.0e-3, q2_max, n_points)
M_V_ref = MV_STD if reference_mode.startswith("Normalización publicada") else M_V
gd_ref = dipole_gd(Q2, M_V_ref)

galster_raw = galster_sachs(Q2, M_V=M_V, lambda_n=lambda_n)
galster_ratio = ratios_from_sachs(galster_raw, gd_ref)

gkex_ratio = None
gkex_error = None
if model_choice in {"GKeX", "Ambas"}:
    try:
        gkex_raw = gkex_sachs(Q2)
        if gkex_raw is not None:
            gkex_ratio = ratios_from_sachs(gkex_raw, gd_ref)
        else:
            gkex_error = (
                "No se encontró `src/form_factors_gkex.py` en el proyecto actual. "
                "La app sigue funcionando para Galster y quedará preparada para GKeX en cuanto ese módulo esté disponible."
            )
    except Exception as exc:  # pragma: no cover - visual fallback
        gkex_error = str(exc)

points_df = load_points_dataframe(uploaded_csv) if show_data else None


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)
axes_map = {
    "GEp": axes[0, 0],
    "GMp": axes[0, 1],
    "GEn": axes[1, 0],
    "GMn": axes[1, 1],
}

y_limits = {
    "GEp": (0.0, 1.2),
    "GMp": (0.90, 1.12),
    "GEn": (-0.05, 0.80),
    "GMn": (0.60, 1.15),
}

for panel in PANEL_ORDER:
    ax = axes_map[panel]

    if model_choice in {"Galster", "Ambas"}:
        ax.plot(Q2, galster_ratio[panel], label="Galster", linewidth=2.0)

    if model_choice in {"GKeX", "Ambas"} and gkex_ratio is not None:
        ax.plot(Q2, gkex_ratio[panel], label="GKeX", linewidth=2.0)

    if points_df is not None:
        sub = points_df[points_df["panel"] == panel]
        if len(sub) > 0:
            if sub["yerr"].notna().any():
                ax.errorbar(
                    sub["Q2"],
                    sub["y"],
                    yerr=sub["yerr"],
                    fmt=".",
                    capsize=2,
                    linestyle="none",
                    label="Datos" if panel == "GEp" else None,
                )
            else:
                ax.plot(sub["Q2"], sub["y"], ".", label="Datos" if panel == "GEp" else None)

    ax.set_ylabel(PANEL_LABELS[panel])
    ax.set_xlabel(r"$|Q^2|\;({\rm GeV}^2)$")
    ax.set_ylim(*y_limits[panel])
    if show_logx:
        ax.set_xscale("log")
    ax.grid(alpha=0.25)

handles, labels = axes[0, 0].get_legend_handles_labels()
if handles:
    axes[0, 0].legend(frameon=False)

st.pyplot(fig, use_container_width=True)

if gkex_error is not None:
    st.warning(gkex_error)


# -----------------------------------------------------------------------------
# Interpretation panel
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Lectura física rápida")
    st.markdown(
        r"- Si aumentas **$M_V$**, el dipolo cae más lentamente y la curva de Galster se endurece.  \n"
        r"- Si aumentas **$\lambda_n$**, el término $\xi_n=(1+\lambda_n\tau)^{-1}$ suprime más deprisa a $G_E^n$.  \n"
        r"- La comparación con **GKeX** sirve para visualizar hasta qué punto un modelo VMD más rico se separa del ansatz dipolar simple."
    )

with col2:
    st.subheader("Propósito didáctico de la simulación")
    st.markdown(
        "Esta visualización está pensada para distinguir tres ideas:  \n"
        "1. qué fija la normalización en $Q^2\to 0$,  \n"
        "2. qué controla la forma funcional de Galster, y  \n"
        "3. por qué GKeX actúa como una referencia fenomenológica más completa en el sector vectorial."
    )


# -----------------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------------
if show_summary_table:
    st.subheader("Valores destacados")
    q2_marks = np.array([0.1, 0.3, 0.5, 0.8, 1.0])
    q2_marks = q2_marks[q2_marks <= q2_max + 1e-12]

    gal_mark = galster_sachs(q2_marks, M_V=M_V, lambda_n=lambda_n)
    gal_ratio_mark = ratios_from_sachs(gal_mark, dipole_gd(q2_marks, M_V_ref))

    gk_ratio_mark = None
    if gkex_ratio is not None:
        gk = gkex_sachs(q2_marks)
        if gk is not None:
            gk_ratio_mark = ratios_from_sachs(gk, dipole_gd(q2_marks, M_V_ref))

    records: list[dict[str, float | str]] = []
    for i, qv in enumerate(q2_marks):
        row = {"Q2 [GeV^2]": float(qv)}
        for panel in PANEL_ORDER:
            row[f"Galster {panel}"] = float(gal_ratio_mark[panel][i])
        if gk_ratio_mark is not None:
            for panel in PANEL_ORDER:
                row[f"GKeX {panel}"] = float(gk_ratio_mark[panel][i])
        records.append(row)

    st.dataframe(pd.DataFrame.from_records(records), use_container_width=True)


# -----------------------------------------------------------------------------
# Footer notes
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Sugerencia de integración con el proyecto: guarda este archivo como "
    "`apps/simulations/app3_em_form_factors.py`. Si quieres mostrar los puntos verdes, añade un CSV en "
    "`refs/em_form_factors/ff_points_green.csv` con columnas `panel,Q2,y,yerr`."
)
