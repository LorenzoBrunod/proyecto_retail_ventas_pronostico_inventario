from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# ========= Utilidades de formato / guardado =========

def _fmt_miles_es(x: float, dec: int = 2) -> str:
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def abrevia_k(x: float, es_dinero: bool = False) -> str:
    if np.isnan(x):
        return "0"
    base = _fmt_miles_es(x / 1000.0, 2) + " k"
    return f"${base}" if es_dinero else base


def _annotate_bars(ax: plt.Axes, bars: List[Any], es_dinero: bool, fontsize: int = 8) -> None:
    for b in bars:
        h = b.get_height()
        if h is None:
            continue
        ax.text(
            b.get_x() + b.get_width() / 2,
            h,
            abrevia_k(float(h), es_dinero=es_dinero),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def _safe_savefig(path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


# ========= Carga de datos / preparación =========

@dataclass
class Config:
    excel_file: Path
    sheet_ordenes: str
    sheet_detalle: Optional[str]
    figs_dir: Path
    years_bcg_train: List[int]
    years_view: List[int]
    qty_forecast_year: int
    qty_train_years: List[int]
    random_state: int


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_retail_data(excel_path: Path, sheet_ordenes: str, sheet_detalle: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if not excel_path.exists():
        raise FileNotFoundError(f"No existe el Excel: {excel_path}")

    xls = pd.ExcelFile(excel_path)
    if sheet_ordenes not in xls.sheet_names:
        raise ValueError(f"No encuentro la hoja '{sheet_ordenes}'. Hojas disponibles: {xls.sheet_names}")

    df_ordenes = pd.read_excel(excel_path, sheet_name=sheet_ordenes)
    df_ordenes = _ensure_datetime(df_ordenes, "Fecha")

    if "Total_Compra" not in df_ordenes.columns:
        if "total compra convertida" in df_ordenes.columns:
            df_ordenes["Total_Compra"] = df_ordenes["total compra convertida"]
        else:
            raise ValueError("No encuentro columna de ingreso (ej: 'Total_Compra' o 'total compra convertida').")

    df_detalle = None
    if sheet_detalle:
        if sheet_detalle not in xls.sheet_names:
            print(f"⚠️  Hoja '{sheet_detalle}' no existe. BCG por producto se omitirá.")
        else:
            df_detalle = pd.read_excel(excel_path, sheet_name=sheet_detalle)

    return df_ordenes, df_detalle


# ========= BCG =========

def bcg_umbral_desde_train(df_ordenes: pd.DataFrame, years: List[int]) -> Dict[str, float]:
    d = df_ordenes.copy()
    d = d[df_ordenes["Fecha"].dt.year.isin(years)].copy()
    if d.empty:
        raise ValueError("No hay datos en los años de entrenamiento para BCG.")

    umbral_ingreso = float(d["Total_Compra"].median())
    if "Cantidad" in d.columns:
        umbral_cantidad = float(d["Cantidad"].mean())
    else:
        umbral_cantidad = 1.0

    return {"umbral_ingreso": umbral_ingreso, "umbral_cantidad": umbral_cantidad}


def bcg_agrega_por_producto(df_ordenes: pd.DataFrame, df_detalle: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_detalle is None:
        if "ID_Producto" in df_ordenes.columns:
            g = df_ordenes.groupby("ID_Producto", dropna=False).agg(
                Cantidad=("ID_Producto", "count"),
                Ingreso_Item=("Total_Compra", "sum"),
            )
            g.reset_index(inplace=True)
            return g
        raise ValueError("No hay detalle y tampoco 'ID_Producto' en Órdenes: no puedo construir BCG por producto.")

    det = df_detalle.copy()

    col_id_prod = None
    for c in det.columns:
        if c.lower() in {"id_producto", "idproducto", "producto", "sku"}:
            col_id_prod = c
            break
    if not col_id_prod:
        raise ValueError("No encuentro columna de producto en detalle (por ej. 'ID_Producto').")

    col_cant = None
    for c in det.columns:
        if c.lower() in {"cantidad", "qty", "unidades"}:
            col_cant = c
            break

    col_subtotal = None
    for c in det.columns:
        if c.lower() in {"subtotal", "importe", "total_linea"}:
            col_subtotal = c
            break

    if col_subtotal:
        det["Ingreso_Item"] = det[col_subtotal]
    else:
        col_price = None
        for c in det.columns:
            if c.lower() in {"precio_unitario", "precio", "p_unit"}:
                col_price = c
                break
        if col_price and col_cant:
            det["Ingreso_Item"] = pd.to_numeric(det[col_price], errors="coerce").fillna(0.0) * pd.to_numeric(det[col_cant], errors="coerce").fillna(0.0)
        else:
            det["Ingreso_Item"] = 0.0

    if col_cant is None:
        det["Cantidad"] = 1.0
    else:
        det["Cantidad"] = pd.to_numeric(det[col_cant], errors="coerce").fillna(0.0)

    g = (
        det.groupby(col_id_prod, dropna=False)
        .agg(Cantidad=("Cantidad", "sum"), Ingreso_Item=("Ingreso_Item", "sum"))
        .reset_index()
        .rename(columns={col_id_prod: "ID_Producto"})
    )
    return g


def bcg_clasificar(df_prod: pd.DataFrame, umbrales: Dict[str, float]) -> pd.DataFrame:
    ui = umbrales["umbral_ingreso"]
    uc = umbrales["umbral_cantidad"]

    def _clase(row: pd.Series) -> str:
        c = float(row["Cantidad"])
        i = float(row["Ingreso_Item"])
        if c >= uc and i >= ui:
            return "Estrella ⭐"
        if c < uc and i >= ui:
            return "Vaca Lechera 🐄"
        if c >= uc and i < ui:
            return "Perro 🐶"
        return "Hormiga 🐜"

    d = df_prod.copy()
    d["Clase_BCG"] = d.apply(_clase, axis=1)
    return d


def plot_bcg_conteo(df_bcg: pd.DataFrame, out_path: Path) -> None:
    conteo = df_bcg["Clase_BCG"].value_counts().reindex(
        ["Estrella ⭐", "Vaca Lechera 🐄", "Perro 🐶", "Hormiga 🐜"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(conteo.index, conteo.values, color="#4c78a8")
    _annotate_bars(ax, list(bars), es_dinero=False, fontsize=8)
    ax.set_title("Conteo por clase BCG")
    ax.set_ylabel("Cantidad de productos")
    _safe_savefig(out_path)


def plot_bcg_totales(df_bcg: pd.DataFrame, out_path: Path) -> None:
    totales = df_bcg.groupby("Clase_BCG", dropna=False)["Ingreso_Item"].sum().reindex(
        ["Estrella ⭐", "Vaca Lechera 🐄", "Perro 🐶", "Hormiga 🐜"], fill_value=0.0
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(totales.index, totales.values, color="#a06cd5")
    _annotate_bars(ax, list(bars), es_dinero=True, fontsize=8)
    ax.set_title("Totales por clase BCG")
    ax.set_ylabel("Ventas")
    _safe_savefig(out_path)


def imprimir_resumen_bcg(df_bcg: pd.DataFrame) -> None:
    conteo = df_bcg["Clase_BCG"].value_counts().reindex(
        ["Estrella ⭐", "Vaca Lechera 🐄", "Perro 🐶", "Hormiga 🐜"], fill_value=0
    )
    totales = df_bcg.groupby("Clase_BCG", dropna=False)["Ingreso_Item"].sum().reindex(
        ["Estrella ⭐", "Vaca Lechera 🐄", "Perro 🐶", "Hormiga 🐜"], fill_value=0.0
    )
    print("\n📦 Resumen BCG (productos):")
    for k in conteo.index:
        c = int(conteo.loc[k])
        t = float(totales.loc[k])
        print(f"• {k}: {c} productos | Ventas: {abrevia_k(t, es_dinero=True)}")
    print(f"• Total filas: {len(df_bcg)}")


# ========= Estacionalidad y outliers =========

def estacionalidad_mensual(df_ordenes: pd.DataFrame, out_path: Path, years: List[int]) -> None:
    d = df_ordenes.copy()
    d = d[d["Fecha"].dt.year.isin(years)].copy()
    d["ym"] = d["Fecha"].dt.to_period("M")
    g = d.groupby("ym", dropna=False)["Total_Compra"].sum().reset_index()
    g["ym"] = g["ym"].astype(str)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(g["ym"], g["Total_Compra"], color="#4c78a8")
    _annotate_bars(ax, list(bars), es_dinero=True, fontsize=8)
    ax.set_title(f"Estacionalidad mensual ({years})")
    ax.set_ylabel("Ventas")
    ax.set_xlabel("Mes")
    ax.tick_params(axis="x", labelrotation=45)
    _safe_savefig(out_path)


def estacionalidad_semanal(df_ordenes: pd.DataFrame, out_path: Path, years: List[int]) -> None:
    d = df_ordenes.copy()
    d = d[d["Fecha"].dt.year.isin(years)].copy()
    d["dow"] = d["Fecha"].dt.dayofweek
    g = d.groupby("dow", dropna=False)["Total_Compra"].sum().reindex(range(7), fill_value=0.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar([str(i) for i in g.index], g.values, color="#72b7b2")
    _annotate_bars(ax, list(bars), es_dinero=True, fontsize=8)
    ax.set_title(f"Estacionalidad semanal ({years})")
    ax.set_ylabel("Ventas")
    _safe_savefig(out_path)


def outliers_estacionales_por_anios(df_ordenes: pd.DataFrame, out_path: Path, years: Tuple[int, int]) -> None:
    d = df_ordenes.copy()
    d = d[d["Fecha"].dt.year.isin(list(years))].copy()
    d["anio"] = d["Fecha"].dt.year
    d["dia"] = d["Fecha"].dt.dayofyear
    g = d.groupby(["anio", "dia"], dropna=False)["Total_Compra"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#4c78a8", "#f58518"]
    for i, y in enumerate(years):
        sub = g[g["anio"] == y]
        if sub.empty:
            continue
        ax.plot(sub["dia"], sub["Total_Compra"], label=str(y), color=colors[i % len(colors)])
        imax = sub["Total_Compra"].idxmax()
        imin = sub["Total_Compra"].idxmin()
        if pd.notna(imax):
            ax.scatter([sub.loc[imax, "dia"]], [sub.loc[imax, "Total_Compra"]], color="red")
        if pd.notna(imin):
            ax.scatter([sub.loc[imin, "dia"]], [sub.loc[imin, "Total_Compra"]], color="red")

    ax.set_title("Outliers estacionales (picos por año)")
    ax.set_xlabel("Día del año"); ax.set_ylabel("Ventas")
    ax.legend()
    _safe_savefig(out_path)


# ========= Comparativas Real vs Pred (mensual) =========

def _join_real_pred(df: pd.DataFrame, y_real: str, y_pred: str) -> pd.DataFrame:
    d = df[["Fecha", y_real, y_pred]].copy()
    d = d.sort_values("Fecha").reset_index(drop=True)
    d["ym"] = d["Fecha"].dt.to_period("M")
    g = d.groupby("ym", dropna=False)[[y_real, y_pred]].sum().reset_index()
    g["ym"] = g["ym"].astype(str)
    return g


def grafico_real_vs_pred_mensual_barras(
    df: pd.DataFrame,
    out_path: Path,
    y_real: str,
    y_pred: str,
    titulo: str,
    es_dinero: bool,
) -> None:
    g = _join_real_pred(df, y_real, y_pred)
    x = np.arange(len(g))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    r1 = ax.bar(x - w / 2, g[y_real].values, width=w, label="Real", color="#4c78a8")
    r2 = ax.bar(x + w / 2, g[y_pred].values, width=w, label="Predicho", color="#f58518")
    _annotate_bars(ax, list(r1), es_dinero=es_dinero, fontsize=8)
    _annotate_bars(ax, list(r2), es_dinero=es_dinero, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(g["ym"], rotation=45)
    ax.set_title(titulo); ax.set_ylabel("Ventas" if es_dinero else "Cantidad")
    ax.legend()
    _safe_savefig(out_path)


# ========= Correlación =========

def correlacion_heatmap(df: pd.DataFrame, out_path: Path, cols: Optional[List[str]] = None) -> None:
    d = df.copy()
    if cols is not None:
        d = d[cols].copy()
    num = d.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        print("⚠️  No hay columnas numéricas para correlación.")
        return
    corr = num.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(corr.shape[1]))
    ax.set_yticks(np.arange(corr.shape[1]))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlación (numérica)")
    _safe_savefig(out_path)


# ========= Modelado (comparación simple) =========

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["year"] = d["Fecha"].dt.year
    d["month"] = d["Fecha"].dt.month
    d["dow"] = d["Fecha"].dt.dayofweek
    d["week"] = d["Fecha"].dt.isocalendar().week.astype(int)
    d["day"] = d["Fecha"].dt.day
    return d


def _cv_score_reg(df: pd.DataFrame, y_col: str, reg: Any) -> Tuple[float, float]:
    d = _add_time_features(df)
    x_mat = d[["year", "month", "dow", "week", "day"]].values
    y_vec = d[y_col].values
    tscv = TimeSeriesSplit(n_splits=3)
    maes: List[float] = []
    r2s: List[float] = []
    for tr, te in tscv.split(x_mat):
        reg_ = reg
        reg_.fit(x_mat[tr], y_vec[tr])
        p = reg_.predict(x_mat[te])
        maes.append(mean_absolute_error(y_vec[te], p))
        r2s.append(r2_score(y_vec[te], p))
    return float(np.mean(maes)), float(np.mean(r2s))


def _fit_best_regressor(df: pd.DataFrame, y_col: str, random_state: int) -> Tuple[Any, Dict[str, float]]:
    d = _add_time_features(df)
    x_mat = d[["year", "month", "dow", "week", "day"]].values
    y_vec = d[y_col].values

    candidatos: Dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
    }
    try:
        from xgboost import XGBRegressor  # type: ignore
        candidatos["XGBRegressor"] = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9, random_state=random_state
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor  # type: ignore
        candidatos["LGBMRegressor"] = LGBMRegressor(
            n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8, random_state=random_state
        )
    except ImportError:
        pass

    maes: Dict[str, float] = {}
    r2s: Dict[str, float] = {}
    best_name = ""
    best_mae = float("inf")
    best_reg: Any = None

    for name, reg in candidatos.items():
        mae, r2 = _cv_score_reg(df, y_col, reg)
        maes[name] = mae
        r2s[name] = r2
        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_reg = reg

    best_reg.fit(x_mat, y_vec)
    print(f"✅ Mejor modelo para '{y_col}': {best_name} | MAE={best_mae:.2f} | R2={r2s[best_name]:.3f}")
    return best_reg, {"MAE": maes[best_name], "R2": r2s[best_name], "Modelo": best_name}


def run_models_comparison_pipeline(
    df_ordenes: pd.DataFrame,
    _figs_dir: Path,
    out_dir: Path,
    random_state: int,
) -> Tuple[Any, Dict[str, float], Any, Dict[str, float]]:
    d = df_ordenes.copy()
    d = d.sort_values("Fecha")

    g_ing = d.groupby(d["Fecha"].dt.date, dropna=False)["Total_Compra"].sum().reset_index()
    g_ing = g_ing.rename(columns={"Fecha": "Fecha", "Total_Compra": "y_ing"})
    g_ing["Fecha"] = pd.to_datetime(g_ing["Fecha"])

    if "Cantidad" in d.columns:
        g_qty = d.groupby(d["Fecha"].dt.date, dropna=False)["Cantidad"].sum().reset_index()
    else:
        g_qty = d.groupby(d["Fecha"].dt.date, dropna=False)["Total_Compra"].size().reset_index(name="Cantidad")
    g_qty = g_qty.rename(columns={"Fecha": "Fecha", "Cantidad": "y_qty"})
    g_qty["Fecha"] = pd.to_datetime(g_qty["Fecha"])

    best_ing, metr_ing = _fit_best_regressor(g_ing.rename(columns={"y_ing": "y"}), "y", random_state)
    best_qty, metr_qty = _fit_best_regressor(g_qty.rename(columns={"y_qty": "y"}), "y", random_state)

    lines = [
        "# Informe de Modelos",
        "Se compararon múltiples modelos con validación temporal (TimeSeriesSplit=3).",
        f"**Ingresos diarios** → Mejor: {metr_ing['Modelo']} | MAE={metr_ing['MAE']:.2f} | R2={metr_ing['R2']:.3f}",
        f"**Cantidad diaria** → Mejor: {metr_qty['Modelo']} | MAE={metr_qty['MAE']:.2f} | R2={metr_qty['R2']:.3f}",
        "",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reporte_modelos.md").write_text("\n".join(lines), encoding="utf-8")

    return best_ing, metr_ing, best_qty, metr_qty


def forecast_quantity_year(df_ordenes: pd.DataFrame, best_qty: Any, year: int, figs_dir: Path) -> pd.DataFrame:
    _ = _add_time_features(df_ordenes.copy())  # mantiene simetría, no se usa directamente
    dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
    cal = pd.DataFrame({"Fecha": dates})
    cal = _add_time_features(cal)

    x_fore = cal[["year", "month", "dow", "week", "day"]].values
    yhat = best_qty.predict(x_fore)
    cal["y_pred"] = yhat

    cal["ym"] = cal["Fecha"].dt.to_period("M")
    g = cal.groupby("ym", dropna=False)["y_pred"].sum().reset_index()
    g["ym"] = g["ym"].astype(str)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(g["ym"], g["y_pred"], marker="o")
    for i, v in enumerate(g["y_pred"].values):
        ax.text(i, v, abrevia_k(float(v), es_dinero=False), ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Pronóstico mensual de Cantidad {year}")
    ax.set_ylabel("Cantidad"); ax.set_xlabel("Mes")
    ax.tick_params(axis="x", labelrotation=45)
    _safe_savefig(figs_dir / f"qty_forecast_{year}.png")

    return cal[["Fecha", "y_pred"]].copy()


# ========= Orquestador por config =========

def run_from_config(config_path: str = "retail_config.json") -> None:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        alt = Path(".retail_config.json")
        if alt.exists():
            cfg_path = alt
        else:
            raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")

    cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = Config(
        excel_file=Path(cfg_json["excel_file"]),
        sheet_ordenes=str(cfg_json.get("sheet_ordenes", "Órdenes")),
        sheet_detalle=cfg_json.get("sheet_detalle", "Detalle_Orden"),
        figs_dir=Path(cfg_json.get("figs_dir", "salidas/figuras")),
        years_bcg_train=list(cfg_json.get("years_bcg_train", [2023])),
        years_view=list(cfg_json.get("years_view", [2023, 2024])),
        qty_forecast_year=int(cfg_json.get("qty_forecast_year", 2025)),
        qty_train_years=list(cfg_json.get("qty_train_years", [2023, 2024])),
        random_state=int(cfg_json.get("random_state", 42)),
    )

    df_ordenes, df_detalle = load_retail_data(cfg.excel_file, cfg.sheet_ordenes, cfg.sheet_detalle)

    print(f"🔎 Excel: {cfg.excel_file.name}")
    print(list(df_ordenes.columns))
    print("\nPrimeras filas de Órdenes:")
    print(df_ordenes.head(5), "\n")

    # ======= BCG =======
    try:
        umbrales = bcg_umbral_desde_train(df_ordenes, cfg.years_bcg_train)
        df_prod = bcg_agrega_por_producto(df_ordenes, df_detalle)
        df_bcg = bcg_clasificar(df_prod, umbrales)

        imprimir_resumen_bcg(df_bcg)

        figs = cfg.figs_dir
        plot_bcg_conteo(df_bcg, figs / "bcg_conteo.png")
        plot_bcg_totales(df_bcg, figs / "bcg_totales.png")

        (figs.parent / "bcg_clasificacion_por_producto.csv").write_text(
            df_bcg.to_csv(index=False), encoding="utf-8"
        )
    except (ValueError, KeyError) as e:
        print(f"⚠️  BCG omitido: {e}")

    # ======= Estacionalidad y outliers =======
    estacionalidad_mensual(df_ordenes, cfg.figs_dir / "estacionalidad_mensual.png", years=cfg.years_view)
    estacionalidad_semanal(df_ordenes, cfg.figs_dir / "estacionalidad_semanal.png", years=cfg.years_view)
    if len(cfg.years_view) >= 2:
        outliers_estacionales_por_anios(
            df_ordenes, cfg.figs_dir / f"outliers_estacionales_{cfg.years_view[0]}_{cfg.years_view[1]}.png",
            years=(cfg.years_view[0], cfg.years_view[1])
        )

    # ======= Modelos y comparativas =======
    best_ing, metr_ing, best_qty, metr_qty = run_models_comparison_pipeline(
        df_ordenes=df_ordenes,
        _figs_dir=cfg.figs_dir,
        out_dir=cfg.figs_dir.parent,
        random_state=cfg.random_state,
    )

    d = _add_time_features(df_ordenes.copy())
    x_all = d[["year", "month", "dow", "week", "day"]].values
    df_ordenes["y_pred_ing"] = best_ing.predict(x_all)

    if "Cantidad" not in df_ordenes.columns:
        df_ordenes["Cantidad"] = 1.0
    df_ordenes["y_pred_qty"] = best_qty.predict(x_all)

    grafico_real_vs_pred_mensual_barras(
        df=df_ordenes.rename(columns={"Total_Compra": "y_ing"}),
        out_path=cfg.figs_dir / "ing_real_vs_pred_mensual_barras_enrq.png",
        y_real="y_ing",
        y_pred="y_pred_ing",
        titulo="Ingresos mensual: Real vs Pred (enriq.)",
        es_dinero=True,
    )

    grafico_real_vs_pred_mensual_barras(
        df=df_ordenes.rename(columns={"Cantidad": "y_qty"}),
        out_path=cfg.figs_dir / "qty_mensual_barras_enrq.png",
        y_real="y_qty",
        y_pred="y_pred_qty",
        titulo="Mensual: Cantidad Real vs Pred (enriq.)",
        es_dinero=False,
    )

    # ======= Pronóstico cantidad 2025 =======
    forecast_quantity_year(df_ordenes, best_qty, cfg.qty_forecast_year, cfg.figs_dir)

    # ======= Correlación =======
    correlacion_heatmap(
        df_ordenes[["Total_Compra", "y_pred_ing", "Cantidad", "y_pred_qty"]].copy(),
        cfg.figs_dir / "correlacion_basica.png",
    )

    print("\n✅ Proceso finalizado. Revisa la carpeta:", cfg.figs_dir.parent)































































