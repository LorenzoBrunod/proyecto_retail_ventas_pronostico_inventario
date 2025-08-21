import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import seaborn as sns
import Funciones

print("Base_Empresa_BigData_Limpio.xlsx")
df = pd.read_excel("Base_Empresa_BigData_Limpio.xlsx")
print(pd.ExcelFile("Base_Empresa_BigData_Limpio.xlsx").sheet_names)
df_envios = pd.read_excel("Base_Empresa_BigData_Limpio.xlsx", sheet_name="Envíos")
print(df_envios.columns)
df_ordenes = pd.read_excel("Base_Empresa_BigData_Limpio.xlsx", sheet_name="Órdenes")
Funciones.mostrar_informacion(df_ordenes)
Funciones.mediana_o_medias(df_ordenes, "Total_Compra", "día")
Funciones.mediana_o_medias(df_ordenes,"Total_Compra","día","Método_Pago")

Funciones.grafico_plot_promedio_con_valores_por_range("Método_Pago", "Total_Compra", "bar", "Promedios por metodo de pago Año 2023-2024", "Promedio de ventas", df_ordenes, plt, [2023, 2024])
Funciones.grafico_estacionalidad_ventas_mes_por_agno_rango(df_ordenes, "Fecha", "Total_Compra", "Estacionalidad por mes Año", plt, [2023, 2024])
Funciones.grafico_estacionalidad_por_dia_semana_range(df_ordenes, "Fecha", "Total_Compra", "Análisis Estacional de Ventas Semanales Año 2023-2024", plt, [2023, 2024])
Funciones.grafico_estacionalidad_semanal_por_agno_rango(df_ordenes, "Fecha", "Total_Compra", [2023, 2024], "Análisis Estacional de Ventas por Semana Año 2023-2024", plt)
Funciones.grafico_estacionalidad_diaria_por_anio_rango(df_ordenes, "Fecha", "Total_Compra", [2023, 2024], "Análisis Estacional Diario de Ventas Año 2023-2024", plt)

eventos = pd.DataFrame({
    "Fecha": [
        "14-02-2023", "18-09-2023", "24-11-2023",
        "25-12-2023", "14-02-2024", "18-09-2024",
        "24-11-2024", "25-12-2024"
    ],
    "Evento": [
        "San Valentín", "Fiestas Patrias", "Black Friday",
        "Navidad", "San Valentín", "Fiestas Patrias",
        "Black Friday", "Navidad"
    ]
})

Funciones.grafico_comparacion_estacional_dias(df_ordenes, "Fecha", "Total_Compra", [2023, 2024], "Comparación Estacional de Ventas Diarias entre 2023 y 2024",eventos, mostrar_texto_outliers=True,guardar_png=False,nombre_archivo=None,abreviar_valores=True,)
outliers_df = Funciones.grafico_comparacion_estacional_dias(df_ordenes, "Fecha", "Total_Compra", [2023, 2024], "Comparación Estacional de Ventas Diarias entre 2023 y 2024",eventos, mostrar_texto_outliers=True,guardar_png=False,nombre_archivo=None,abreviar_valores=True)
print(outliers_df)

Funciones.histograma_seoborn_id_repetidos(df_ordenes, "ID_Cliente", "ID_Orden", 20, False, "Distribución de Frecuencia de Compra Año 2023-2024", "Compras por Cliente", "Cantidad de Clientes", sns, plt, [2023, 2024], "Fecha")
Funciones.tipo_de_distribucion(Funciones, shapiro, df_ordenes, "Total_Compra", [2023, 2024], "Fecha")

print(df_ordenes["Método_Pago"].value_counts(dropna=False))
Funciones.categorizar_columnas(df_ordenes, "Método_Pago", "Método_Pago_Estandarizado", "Tarjeta", "Transferencia", "Efectivo", "App móvil")
print(Funciones.mostrar_primeras_filas(df_ordenes, 1))
df_ordenes["Día del Año"] = pd.to_datetime(df_ordenes["Fecha"]).dt.dayofyear
Funciones.matriz_correlacion(df_ordenes, "Total_Compra", "Método_Pago_Estandarizado", "Día del Año", [2023, 2024], "Fecha")
Funciones.mapa_correlaciomapa_correlacion_por_agnio(df_ordenes, "Total_Compra", "Método_Pago_Estandarizado", "Día del Año", plt, sns, "coolwarm", "Mapa de correlación Año", [2023, 2024], "Fecha")

# Cargar las hojas necesarias
df_detalle = pd.read_excel("Base_Empresa_BigData_Limpio.xlsx", sheet_name="Detalle_Orden")
df_producto = pd.read_excel("Base_Empresa_BigData_Limpio.xlsx", sheet_name="Productos")
print(df_producto.columns)

# Unir Órdenes + Detalle_Orden
df_ordenes_detalle = df_ordenes.merge(df_detalle, on="ID_Orden", how="inner")
Funciones.categorizar_columnas_con_lista_agnos(df_producto, "Categoría", "Categoría_categorizada", ["Oficina", "Hogar", "Juguetería", "Moda", "Herramientas", "Electrónica"])

# Unir el resultado con Producto
df_ordenes_full = df_ordenes_detalle.merge(df_producto, on="ID_Producto", how="left")
print(df_ordenes_full)
#unir "Nombre_Producto_Real"


df_ordenes_full["Ingreso_Item"] = df_ordenes_full["Cantidad"] * df_ordenes_full["Precio"]

print("Las columnas de df_ordenes_full son : \n",df_ordenes_full.columns)

# Gráficos adicionales
Funciones.grafico_plot_promedio_con_valores_por_rango_tiempo("Categoría", "Ingreso_Item", "bar", "Promedio por Categoría Año 2023-2024", "Ingreso promedio", df_ordenes_full, plt, [2023, 2024], "Fecha")
Funciones.grafico_circular_por_rango_tiempo(df_ordenes_full, plt, "Categoría", "Participación por Categoría de Producto Año 2023-2024", [2023, 2024], "Fecha")
Funciones.graficoapiladoconporcentaje_por_agno(df_ordenes_full, sns, plt, "Distribución % de Categoría por Método de Pago Año 2023-2024", "Método_Pago", "Categoría", 6, [2023, 2024], "Fecha")
Funciones.mapa_correlaciomapa_correlacion_por_agnio(df_ordenes_full, "Cantidad", "Precio", "Ingreso_Item", plt, sns, "coolwarm", "Mapa de Correlación entre Precio, Cantidad e Ingreso Año", [2023, 2024], "Fecha")
Funciones.grafico_plot_total("Método_Pago", "Ingreso_Item", "bar", "Total de Ingresos por Método de Pago Años 2023-2024", "Ingreso Total", df_ordenes_full, plt, [2023, 2024], "Fecha")
Funciones.mostrar_primeras_filas(df_producto, 20)

# =================== MODELO BASE (sin nuevas features) =======================
modelo_base, mae_base, r2_base, _ = Funciones.modelo_random_forest_total_compra(
    df_ordenes_full,
    ["Método_Pago_Estandarizado", "Día del Año", "Categoría_categorizada"],
    "Ingreso_Item",
    [2023, 2024],
    "Fecha"
)

df_pred_base = Funciones.predecir_datos_avanzado(
    df_ordenes_full,
    modelo_base,
    ["Método_Pago_Estandarizado", "Día del Año", "Categoría_categorizada"],
    [2023,2024,2025],
    "Fecha",
    "Prediccion_Ingreso_Base"
)

# =================== GENERAR NUEVAS FEATURES =======================
from Funciones import generar_nuevas_features, detectar_meses_estacionales_por_outliers

df_enriquecido = generar_nuevas_features(
    df_ordenes_full,
    eventos_df=eventos,
    meses_estacionales=None,
    detectar_meses_fn=detectar_meses_estacionales_por_outliers,
    anios=[2023, 2024]
)

# Codificar variable categórica BCG
df_enriquecido = pd.get_dummies(df_enriquecido, columns=["BCG_Clasificacion"], drop_first=True)

columnas_enriquecidas = [
    "Método_Pago_Estandarizado", "Día del Año", "Categoría_categorizada",
    "Fin_de_Semana", "Descuento_Aplicado", "Producto_Premium",
    "Dia_Feriado", "Es_Estacional"
] + [col for col in df_enriquecido.columns if col.startswith("BCG_Clasificacion_")]

# =================== MODELO ENRIQUECIDO (con nuevas features) =======================
modelo_enriquecido, mae_e, r2_e, _ = Funciones.modelo_random_forest_total_compra(
    df_enriquecido,
    columnas_enriquecidas,
    "Ingreso_Item",
    [2023, 2024],
    "Fecha"
)

df_pred_enriquecido = Funciones.predecir_datos_avanzado(
    df_enriquecido,
    modelo_enriquecido,
    columnas_enriquecidas,
    [2023,2024,2025],
    "Fecha",
    "Prediccion_Ingreso_Enriquecido"
)

# ====== MÉTRICAS EXTRA (WAPE, sMAPE, MASE, Bias) ======

# 1) Tomar las columnas reales usadas por cada modelo (evita hardcodear)
if hasattr(modelo_base, "feature_names_in_"):
    columnas_base = list(modelo_base.feature_names_in_)
else:
    # fallback: las 3 que usaste al entrenar el BASE
    columnas_base = ["Método_Pago_Estandarizado", "Día del Año", "Categoría_categorizada"]

if hasattr(modelo_enriquecido, "feature_names_in_"):
    columnas_enriq_usadas = list(modelo_enriquecido.feature_names_in_)
else:
    # fallback: la lista enriquecida que ya tienes en tu script
    # (incluye Fin_de_Semana, Descuento_Aplicado, Producto_Premium, Dia_Feriado, Es_Estacional,
    #  y los dummies que empiezan con 'BCG_Clasificacion_')
    columnas_enriq_usadas = columnas_enriquecidas

# 2) Calcular las métricas con el helper del módulo
met_base = Funciones.metricas_modelo(
    modelo_base,
    df_ordenes_full,
    columnas_base,
    "Ingreso_Item",
    (2023, 2024),
    "Fecha"
)
met_e = Funciones.metricas_modelo(
    modelo_enriquecido,
    df_enriquecido,
    columnas_enriq_usadas,
    "Ingreso_Item",
    (2023, 2024),
    "Fecha"
)

print("\n📊 Comparación de desempeño de los modelos:")
print(f"Modelo BASE     - MAE: {mae_base:.2f} | R²: {r2_base:.4f} | "
      f"WAPE: {met_base['WAPE']*100:.2f}% | "
      f"sMAPE: {met_base['sMAPE']*100:.2f}% | "
      f"MASE: {met_base['MASE']:.3f} | "
      f"Bias: {met_base['Bias']:.2f} ({met_base['Bias_pct']*100:.2f}%)")

print(f"Modelo ENRIQ.   - MAE: {mae_e:.2f} | R²: {r2_e:.4f} | "
      f"WAPE: {met_e['WAPE']*100:.2f}% | "
      f"sMAPE: {met_e['sMAPE']*100:.2f}% | "
      f"MASE: {met_e['MASE']:.3f} | "
      f"Bias: {met_e['Bias']:.2f} ({met_e['Bias_pct']*100:.2f}%)")



# =================== GRÁFICOS DE COMPARACIÓN =======================
Funciones.grafico_lineas_ingresos_por_anio(df_pred_base, "Categoría", "Ingreso_Item", "Prediccion_Ingreso_Base", "Línea: Ingreso Real vs Predicho (Base) por Categoría 2023-2024", [2023,2024], "Fecha", plt)
Funciones.grafico_baseball_bat_ingresos_por_anio(df_pred_base, "Método_Pago", "Ingreso_Item", "Prediccion_Ingreso_Base", "Lollipop: Ingreso Real vs Predicho (Base) por Método de Pago 2024-2024", [2023,2024], "Fecha", plt)
Funciones.grafico_lineas_ingresos_por_anio(df_pred_base, "Categoría", "Ingreso_Item", "Prediccion_Ingreso_Base", "Línea: Ingreso Real vs Predicho (Base) por Categoría 2025", [2025], "Fecha", plt)

Funciones.grafico_lineas_ingresos_por_anio(df_pred_enriquecido, "Categoría", "Ingreso_Item", "Prediccion_Ingreso_Enriquecido", "Línea: Ingreso Real vs Predicho (Enriquecido) por Categoría", [2023,2025], "Fecha", plt)
Funciones.grafico_baseball_bat_ingresos_por_anio(df_pred_enriquecido, "Método_Pago", "Ingreso_Item", "Prediccion_Ingreso_Enriquecido", "Lollipop: Ingreso Real vs Predicho (Enriquecido) por Método de Pago", [2023,2024], "Fecha", plt)
Funciones.grafico_lineas_ingresos_por_anio(df_pred_enriquecido, "Categoría", "Ingreso_Item", "Prediccion_Ingreso_Enriquecido", "Línea: Ingreso Real vs Predicho (Enriquecido) por Categoría", [2025], "Fecha", plt)




Funciones.grafico_lineas_ingresos_por_anios_filtro(
    df_pred_enriquecido,
    grupo_col="Categoría",
    real_col="Ingreso_Item",
    pred_col="Prediccion_Ingreso_Enriquecido",
    titulo="Línea: Ingreso Real vs Predicho por Categoría (Mensual)",
    anios=[2023,2024],
    fecha_col="Fecha",
    plt_mod=plt,
    frecuencia="M",categorias_filtrar=["Hogar"]
)



modelo_enriquecido, mae_e, r2_e, _ = Funciones.modelo_random_forest_total_compra(
    df_enriquecido,
    columnas_enriquecidas,
    "Cantidad",
    [2023, 2024],
    "Fecha"
)


df_pred_cantidad = Funciones.predecir_datos_avanzado(
    df_enriquecido,
    modelo_enriquecido,
    columnas_enriquecidas,
    [2023, 2024],
    "Fecha",
    "Prediccion_Cantidad"
)

Funciones.grafico_lineas_ingresos_por_anios_filtro(
    df_pred_cantidad,
    grupo_col="Categoría",
    real_col="Cantidad",
    pred_col="Prediccion_Cantidad",
    titulo="Línea: Cantidad Real vs Predicha por Categoría (Mensual)",
    anios=[2023,2024],
    fecha_col="Fecha",
    plt_mod=plt,
    frecuencia="M",categorias_filtrar=["Hogar"]
)



print("Las columnasd de df_envios :\n",df_envios.columns)

# Agregar Nombre_Producto_Real a df_envios usando ID_Producto
df_envios = df_envios.merge(
    df_ordenes_full[["ID_Orden", "ID_Producto", "Nombre_Producto_Real"]],
    on="ID_Orden",
    how="left"
)





























