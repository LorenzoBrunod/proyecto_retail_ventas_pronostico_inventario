# ============================ Funcioness.py ============================
from __future__ import annotations
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

# ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Estadística
from scipy.stats import shapiro, zscore

# Dash / Plotly
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import textwrap


__all__ = [
    "generar_nuevas_features",
    "detectar_meses_estacionales_por_outliers",
    "modelo_random_forest_total_compra",
    "predecir_datos_avanzado",
    "crear_layout",
    "registrar_callbacks",
]

# ------------------------- Helpers ------------------------------------
def _mean_or_median(series: pd.Series) -> Tuple[float, str]:
    """(valor, 'media'|'mediana') usando Shapiro (p<0.05 -> mediana)."""
    s = pd.to_numeric(pd.Series(series).dropna(), errors="coerce")
    if len(s) < 3:
        return (float(s.mean()) if len(s) else 0.0, "NA")
    try:
        _, p = shapiro(s)
    except Exception:
        p = 1.0
    if p < 0.05:
        return float(s.median()), "mediana"
    return float(s.mean()), "media"


def _fmt_periodo(ts: pd.Timestamp, modo: str) -> str:
    if modo == "Mes":
        return pd.Timestamp(ts).strftime("%m-%Y")
    p = pd.Period(ts, freq="Q")
    return f"Q{p.quarter}-{p.year}"


def _ph(msg: str):
    fig = px.imshow([[0]], text_auto=True, color_continuous_scale="Blues",
                    aspect="auto", title=msg)
    fig.update_layout(template="plotly_white", xaxis_title="", yaxis_title="")
    return fig


def _wrap_index(idx: pd.Index, width: int = 22) -> list[str]:
    """Envuelve etiquetas largas en varias líneas para mejor lectura."""
    return ["<br>".join(textwrap.wrap(str(x), width=width, break_long_words=False)) for x in idx]


def _apply_layout(fig, pivot):
    """Ajusta altura y márgenes para que se vean todas las etiquetas."""
    n_rows = int(len(pivot.index))
    try:
        max_len = int(pd.Series(pivot.index).astype(str).str.len().max())
    except Exception:
        max_len = 15

    fig.update_layout(
        height=max(420, min(1400, 34 * n_rows)),             # ~34 px por fila
        margin=dict(l=max(120, 7 * max_len), r=10, t=40, b=10),
        template="plotly_white",
    )
    fig.update_yaxes(automargin=True, tickfont=dict(size=12))
    fig.update_xaxes(automargin=True, tickangle=-45)
    return fig


# -------------------- Feature engineering -----------------------------
def generar_nuevas_features(
    df: pd.DataFrame,
    eventos_df: pd.DataFrame | None = None,
    meses_estacionales: list[int] | None = None,
    detectar_meses_fn=None,
    anios: list[int] = [2023, 2024],
) -> pd.DataFrame:
    d = df.copy()
    d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
    d = d[d["Fecha"].dt.year.isin(anios + [2025])]

    d["Día del Año"] = d["Fecha"].dt.dayofyear
    d["Fin_de_Semana"] = d["Fecha"].dt.weekday >= 5

    if "Descuento" in d.columns:
        d["Descuento_Aplicado"] = d["Descuento"].fillna(0) > 0
    else:
        d["Descuento_Aplicado"] = False

    if "Precio" in d.columns:
        d["Producto_Premium"] = d["Precio"] > d["Precio"].median()
    else:
        d["Producto_Premium"] = False

    if eventos_df is not None and "Fecha" in eventos_df.columns:
        ev = eventos_df.copy()
        ev["Fecha"] = pd.to_datetime(ev["Fecha"], errors="coerce")
        d = d.merge(ev[["Fecha", "Evento"]], on="Fecha", how="left")
        d["Dia_Feriado"] = d["Evento"].notna()
    else:
        d["Dia_Feriado"] = False

    if (meses_estacionales is None) and (detectar_meses_fn is not None):
        meses_estacionales = detectar_meses_fn(
            df=d,
            columna_fecha="Fecha",
            columna_valor="Ingreso_Item" if "Ingreso_Item" in d.columns else "Cantidad",
            anios=anios,
        )
    d["Es_Estacional"] = d["Fecha"].dt.month.isin(meses_estacionales or [])

    return d


def detectar_meses_estacionales_por_outliers(
    df: pd.DataFrame,
    columna_fecha: str = "Fecha",
    columna_valor: str = "Ingreso_Item",
    anios: list[int] | None = None,
) -> list[int]:
    d = df.copy()
    d[columna_fecha] = pd.to_datetime(d[columna_fecha], errors="coerce")
    if anios:
        d = d[d[columna_fecha].dt.year.isin(anios)]
    d["Mes"] = d[columna_fecha].dt.month
    res = d.groupby("Mes")[columna_valor].sum().sort_index()
    if len(res) < 3:
        return []
    z = np.abs(zscore(res))
    return res.index[(z > 1.5)].tolist()


# -------------------- Modelado y predicción ---------------------------
def modelo_random_forest_total_compra(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    anios_entrenamiento: list[int],
    columna_fecha: str,
):
    d = df.copy()
    d[columna_fecha] = pd.to_datetime(d[columna_fecha], errors="coerce")
    d = d[d[columna_fecha].dt.year.isin(anios_entrenamiento)]

    X = pd.get_dummies(d[features], drop_first=True)
    y = d[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return model, mae, rmse, list(X.columns)


def predecir_datos_avanzado(
    df: pd.DataFrame,
    modelo,
    columnas_features: list[str],
    anios_prediccion: list[int],
    columna_fecha: str,
    nombre_columna_pred: str,
) -> pd.DataFrame:
    d = df.copy()
    d[columna_fecha] = pd.to_datetime(d[columna_fecha], errors="coerce")

    mask = d[columna_fecha].dt.year.isin(anios_prediccion)
    X_pred = pd.get_dummies(d.loc[mask, columnas_features], drop_first=True)
    if hasattr(modelo, "feature_names_in_"):
        X_pred = X_pred.reindex(columns=modelo.feature_names_in_, fill_value=0)
    d.loc[mask, nombre_columna_pred] = modelo.predict(X_pred)
    return d


# ------------------------ Dash: layout --------------------------------
def crear_layout(app: Any, years_opts: List[int], centros_opts: List[str]) -> html.Div:
    """
    Layout con segmentadores y 3 heatmaps.
    Incluye CSS inline para quitar la barra gris de los RangeSlider
    y ocultar puntos/ticks (sin necesidad de carpeta assets).
    """
    # Sanitiza listas
    years_opts   = sorted(set(int(y) for y in years_opts)) if years_opts else []
    centros_opts = sorted(set(str(c) for c in centros_opts)) if centros_opts else []

    # CSS inline (sin assets): quita riel gris y oculta ticks/puntos del slider
    style_block = dcc.Markdown(
        """
        <style>
          /* Quitar barra gris del slider (rail) */
          .slider-clean .rc-slider-rail {
            background-color: transparent !important;
          }
          /* (Opcional) ocultar puntitos/ticks para evitar "borroso" */
          .slider-clean .rc-slider-mark,
          .slider-clean .rc-slider-step,
          .slider-clean .rc-slider-dot,
          .slider-clean .rc-slider-dot-active {
            display: none !important;
          }
          /* Estética clara del track y manillas */
          .slider-clean .rc-slider-track {
            background-color: #2563eb !important;
            height: 8px; border-radius: 9999px;
          }
          .slider-clean .rc-slider-handle {
            width: 18px; height: 18px; margin-top: -5px;
            border: 2px solid #2563eb; background: #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,.15);
          }
          .slider-clean .rc-slider-handle:focus,
          .slider-clean .rc-slider-handle:hover {
            box-shadow: 0 0 0 3px rgba(37,99,235,.25);
            outline: none;
          }
        </style>
        """,
        dangerously_allow_html=True,
    )

    # Panel de filtros
    controls = dbc.Card(
        [
            html.H5("Filtros", className="mb-2"),

            dbc.Label("Año(s)"),
            dcc.Dropdown(
                id="dd_years",
                options=[{"label": str(y), "value": int(y)} for y in years_opts],
                value=years_opts,  # seleccionar todos
                multi=True,
                placeholder="Selecciona años",
            ),

            dbc.Label("Agrupar por"),
            dbc.RadioItems(
                id="rd_periodo",
                options=[{"label": "Mes", "value": "Mes"},
                         {"label": "Trimestre", "value": "Trimestre"}],
                value="Mes",
                inline=True,
            ),

            dbc.Label("Centro(s) de distribución"),
            dcc.Dropdown(
                id="dd_centros",
                options=[{"label": c, "value": c} for c in centros_opts],
                value=centros_opts[:2] if len(centros_opts) > 1 else centros_opts,
                multi=True,
                placeholder="Selecciona centro(s)",
            ),

            html.Hr(className="my-3"),
            html.H6("Filtros Q* / ROP", className="mb-2"),

            dbc.Label("Rango Q*"),
            dcc.RangeSlider(
                id="sl_q", min=0, max=1500, step=10, value=[0, 1500],
                marks=None, dots=False, allowCross=False, updatemode="drag",
                tooltip={"always_visible": False, "placement": "bottom"},
                className="slider-clean",
            ),
            html.Div(id="lbl_q", className="text-muted small mt-1"),

            dbc.Label("Rango Lead Time (días)"),
            dcc.RangeSlider(
                id="sl_lt", min=0, max=60, step=1, value=[0, 60],
                marks=None, dots=False, allowCross=False, updatemode="drag",
                tooltip={"always_visible": False, "placement": "bottom"},
                className="slider-clean",
            ),
            html.Div(id="lbl_lt", className="text-muted small mt-1"),
        ],
        body=True,
    )

    # Panel de resultados
    grids = dbc.Card(
        [
            html.H4("Optimización de Inventarios — Q (EOQ), ROP y Lead Time", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="fig_q"),  md=12),
                    dbc.Col(dcc.Graph(id="fig_rop"), md=12),
                    dbc.Col(dcc.Graph(id="fig_lt"),  md=12),
                ],
                className="g-4",
            ),
        ],
        body=True,
    )

    # Estructura final
    return dbc.Container(
        [
            style_block,  # CSS inline
            dbc.Row(
                [
                    dbc.Col(controls, md=4),
                    dbc.Col(grids,   md=8),
                ],
                className="g-4",
            ),
        ],
        fluid=True,
    )



# --------------------- Dash: callbacks --------------------------------
def registrar_callbacks(app, df_enriquecido: pd.DataFrame, df_envios: pd.DataFrame,
                        costos_orden: dict, costos_mantencion: dict):
    """
    Nombres de columnas esperados:
      - Centro_Distribucion
      - ID_Producto
      - Nombre_Producto_Real
      - Fecha_Envío
      - Fecha_Entrega_Final
    """
    import traceback

    COL_CENTRO = "Centro_Distribucion"
    COL_ID     = "ID_Producto"
    COL_NOMBRE = "Nombre_Producto_Real"
    COL_FENV   = "Fecha_Envío"
    COL_FENT   = "Fecha_Entrega_Final"

    # ---- Normalización de ENVÍOS / Lead Time
    faltan_env = [c for c in [COL_FENV, COL_FENT] if c not in df_envios.columns]
    if faltan_env:
        msg = f"Faltan columnas en ENVÍOS: {', '.join(faltan_env)}"

        @app.callback(Output("fig_q","figure"),Output("fig_rop","figure"),Output("fig_lt","figure"),
                      Input("dd_years","value"),Input("rd_periodo","value"),
                      Input("dd_centros","value"),Input("sl_q","value"),Input("sl_lt","value"))
        def _err(*_): 
            return _ph(msg), _ph(msg), _ph(msg)

        @app.callback(Output("lbl_q","children"), Input("sl_q","value"))
        def _lbl_q(_): return "–"

        @app.callback(Output("lbl_lt","children"), Input("sl_lt","value"))
        def _lbl_lt(_): return "–"

        return

    df_env = df_envios.copy()
    df_env[COL_FENV] = pd.to_datetime(df_env[COL_FENV], errors="coerce")
    df_env[COL_FENT] = pd.to_datetime(df_env[COL_FENT], errors="coerce")
    df_env = df_env[df_env[COL_FENV].notna() & df_env[COL_FENT].notna()]
    df_env["Lead_Time_Dias"] = (df_env[COL_FENT] - df_env[COL_FENV]).dt.days

    # Si en envíos falta nombre pero hay ID, lo traemos de ventas
    if (COL_NOMBRE not in df_env.columns) and (COL_ID in df_env.columns) \
       and {COL_ID, COL_NOMBRE}.issubset(df_enriquecido.columns):
        df_env = df_env.merge(
            df_enriquecido[[COL_ID, COL_NOMBRE]].drop_duplicates(),
            on=COL_ID, how="left"
        )

    # Etiquetas sliders
    @app.callback(Output("lbl_q","children"), Input("sl_q","value"))
    def _lbl_q(v):
        try: return f"{int(v[0])} – {int(v[1])}"
        except: return "–"

    @app.callback(Output("lbl_lt","children"), Input("sl_lt","value"))
    def _lbl_lt(v):
        try: return f"{int(v[0])} d – {int(v[1])} d"
        except: return "–"

    # Callback principal
    @app.callback(
        Output("fig_q","figure"), Output("fig_rop","figure"), Output("fig_lt","figure"),
        Input("dd_years","value"), Input("rd_periodo","value"),
        Input("dd_centros","value"), Input("sl_q","value"), Input("sl_lt","value"),
    )
    def _update(anios_sel, periodo, centros_sel, rng_q, rng_lt):
        try:
            if not anios_sel:
                return _ph("Selecciona año(s)"), _ph("Selecciona año(s)"), _ph("Selecciona año(s)")
            if not centros_sel:
                return _ph("Selecciona centro(s)"), _ph("Selecciona centro(s)"), _ph("Selecciona centro(s)")

            # Normalizar entradas
            if isinstance(anios_sel, (int, float, str)): anios_sel = [int(anios_sel)]
            anios_sel = [int(x) for x in anios_sel]
            if isinstance(centros_sel, str): centros_sel = [centros_sel]
            centros_sel = [str(x) for x in centros_sel]
            try:  rng_q  = [float(rng_q[0]),  float(rng_q[1])]
            except: rng_q = [0.0, 1e9]
            try:  rng_lt = [float(rng_lt[0]), float(rng_lt[1])]
            except: rng_lt = [0.0, 365]

            # Ventas base
            d = df_enriquecido.copy()
            d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
            d = d[d["Fecha"].dt.year.isin(anios_sel)]

            # Asegurar Centro_Distribucion en ventas
            if COL_CENTRO not in d.columns:
                if (COL_ID in d.columns) and (COL_ID in df_env.columns) and (COL_CENTRO in df_env.columns):
                    d = d.merge(df_env[[COL_ID, COL_CENTRO]].drop_duplicates(), on=COL_ID, how="left")
                elif (COL_NOMBRE in d.columns) and (COL_NOMBRE in df_env.columns) and (COL_CENTRO in df_env.columns):
                    d = d.merge(df_env[[COL_NOMBRE, COL_CENTRO]].drop_duplicates(), on=COL_NOMBRE, how="left")

            if COL_CENTRO not in d.columns:
                return _ph(f"No se pudo mapear {COL_CENTRO} a VENTAS"), _ph("—"), _ph("—")

            # Filtro por centros
            d = d[d[COL_CENTRO].isin(centros_sel)].copy()
            if d.empty:
                return _ph("Sin datos para los filtros"), _ph("Sin datos"), _ph("Sin datos")

            # Cantidad efectiva (usa predicción en 2025)
            if "Prediccion_Cantidad" in d.columns:
                y = d["Fecha"].dt.year
                d["Cantidad_Efectiva"] = np.where(
                    (y == 2025) & d["Prediccion_Cantidad"].notna(),
                    d["Prediccion_Cantidad"],
                    d.get("Cantidad", 0)
                )
            else:
                d["Cantidad_Efectiva"] = d.get("Cantidad", 0)

            # Periodo
            d["PeriodoTS"] = d["Fecha"].dt.to_period("M").dt.to_timestamp() if periodo == "Mes" \
                             else d["Fecha"].dt.to_period("Q").dt.to_timestamp()

            prods = d[COL_NOMBRE].dropna().unique().tolist()
            if not prods:
                return _ph("Sin productos tras filtros"), _ph("Sin productos"), _ph("Sin productos")

            # Cálculos
            out = []
            for prod in prods:
                d_prod = d[d[COL_NOMBRE] == prod]
                for periodo_key, df_p in d_prod.groupby("PeriodoTS"):
                    dem_diaria = df_p.groupby(df_p["Fecha"].dt.date)["Cantidad_Efectiva"].sum()
                    D, _ = _mean_or_median(dem_diaria)

                    env_f = df_env[
                        (df_env.get(COL_NOMBRE) == prod) &
                        (df_env[COL_CENTRO].isin(centros_sel)) &
                        (df_env[COL_FENV] <= pd.Timestamp(periodo_key))
                    ]
                    LT, _ = _mean_or_median(env_f["Lead_Time_Dias"])

                    s = float(costos_orden.get(prod, 100))
                    h = float(costos_mantencion.get(prod, 10))
                    Q = np.sqrt(max(2 * D * s / h, 0)) if h > 0 else 0.0
                    ROP = D * LT

                    out.append({"Producto": prod, "Periodo": _fmt_periodo(periodo_key, periodo),
                                "Q": Q, "ROP": ROP, "LT": LT})

            res = pd.DataFrame(out)
            if res.empty:
                return _ph("Sin resultados"), _ph("Sin resultados"), _ph("Sin resultados")

            # Rango sliders
            res = res[(res["Q"].between(rng_q[0], rng_q[1])) & (res["LT"].between(rng_lt[0], rng_lt[1]))]
            if res.empty:
                return _ph("Fuera de rango"), _ph("Fuera de rango"), _ph("Fuera de rango")

            piv_q   = res.pivot_table(index="Producto", columns="Periodo", values="Q",   aggfunc="mean").fillna(0)
            piv_rop = res.pivot_table(index="Producto", columns="Periodo", values="ROP", aggfunc="mean").fillna(0)
            piv_lt  = res.pivot_table(index="Producto", columns="Periodo", values="LT",  aggfunc="mean").fillna(0)

            # Envolver etiquetas para que entren
            piv_q.index   = _wrap_index(piv_q.index,   width=22)
            piv_rop.index = _wrap_index(piv_rop.index, width=22)
            piv_lt.index  = _wrap_index(piv_lt.index,  width=22)

            # Orden temporal de columnas
            def _order(cols: pd.Index) -> list:
                if len(cols) == 0: return list(cols)
                if "Q" in str(cols[0]):  # formato Q-YYYY
                    keys = [(int(c.split("-")[1]), int(c.split("-")[0][1])) for c in cols]
                    return [c for _, c in sorted(zip(keys, cols))]
                ts = pd.to_datetime(["01-" + c for c in cols], format="%d-%m-%Y", errors="coerce")
                return [c for _, c in sorted(zip(ts, cols))]

            piv_q, piv_rop, piv_lt = piv_q[_order(piv_q.columns)], piv_rop[_order(piv_rop.columns)], piv_lt[_order(piv_lt.columns)]

            fig_q  = px.imshow(piv_q,  text_auto=".0f", color_continuous_scale="Blues",  aspect="auto", title="Heatmap Q*")
            fig_rop= px.imshow(piv_rop, text_auto=".0f", color_continuous_scale="PuBu",   aspect="auto", title="Heatmap ROP")
            fig_lt = px.imshow(piv_lt, text_auto=".0f", color_continuous_scale="YlGnBu", aspect="auto", title="Heatmap Lead Time")

            fig_q   = _apply_layout(fig_q,   piv_q)
            fig_rop = _apply_layout(fig_rop, piv_rop)
            fig_lt  = _apply_layout(fig_lt,  piv_lt)

            return fig_q, fig_rop, fig_lt

        except Exception as e:
            print("\n[Dash callback error]")
            import traceback; traceback.print_exc()
            fig = _ph(f"⚠️ {type(e).__name__}: {e}")
            return fig, fig, fig



