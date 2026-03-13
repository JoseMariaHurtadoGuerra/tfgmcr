# Streamlit app: factores de forma EM del nucleón

## Qué incluye

- `app3_em_form_factors.py`: app Streamlit para comparar Galster y GKeX en los cuatro cocientes de la figura.
- `ff_points_green_template.csv`: plantilla mínima para superponer puntos experimentales.

## Dónde colocarlo en tu proyecto

- App: `apps/simulations/app3_em_form_factors.py`
- Datos opcionales: `refs/em_form_factors/ff_points_green.csv`

## Requisito para GKeX

La app intenta importar:

```python
from src.form_factors_gkex import sachs_gkex
```

y acepta varias firmas posibles (`Q2`, `q2`, `Q2_GeV2`). Si tu función devuelve un formato distinto, adapta el bloque `_coerce_gkex_output`.

## Ejecución local

```bash
streamlit run apps/simulations/app3_em_form_factors.py
```

## Diseño físico adoptado

- `M_V` y `lambda_D^V` están ligados por definición: `lambda_D^V = 4 M_N^2 / M_V^2`.
- `lambda_n` es editable porque controla la forma de `G_E^n`.
- GKeX aparece como curva de referencia sin sliders internos.
- Por defecto, la normalización del dipolo de referencia es la publicada en la tesis: `M_V^ref = 0.843 GeV`.
