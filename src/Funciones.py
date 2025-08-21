from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


def contador_sexos(lista):
    contador={"Masculino":0,"Femenino":0}
    for _,x in lista.iterrows():
       valor =str(x["sexo"])
       if valor in contador:
            contador[valor]+=1
    return contador

def contador_estados_alumnos(lista):
    contador={"False":0,"True":0}
    for _,x in lista.iterrows():
        value=str(x["repitente"])
        if value in contador:
            contador[value]+=1
    return contador

def ocntador_zona(lista):
    contador={"Rural":0,"Urbana":0}
    for _,x in lista.iterrows():
        vals=str(x["zona"])
        if vals in contador:
            contador[vals]+=1
    return contador


def posicion(n):
    if n==1.0:
        return "primer lugar"
    elif n==2.0:
        return "Segundo lugar"
    elif n==3.0:
        return "Tercer lugar"
    else:
        return "No premiado"


def compartamientodis(p):
     if p >0.5:
         return " Tiene un compartamiento normal\nPuedes ocupar los siguiente metodos paramtricos metodos paramétricos :\nt-test (Student o Welch) : Comparar promedios entre 2 grupos\nANOVA (análisis de varianza) : Comparar promedios entre >2 grupos\nCorrelación de Pearson : Ver relación entre dos variables numéricas\nRegresión lineal: Ajustar una línea o modelo\nt de una muestra : Comparar una muestra con un valor esperado\nF-test o Levene: Comparar varianzas"
     elif p > 0.10:
         return " ◉Normalidad muy probable\nPuedes ocupar los siguiente metodos paramtricos metodos paramétricos :\n\n◉t-test (Student o Welch) : Comparar promedios entre 2 grupos\n◉ANOVA (análisis de varianza) : Comparar promedios entre >2 grupos\n◉Correlación de Pearson : Ver relación entre dos variables numéricas\n◉Regresión lineal: Ajustar una línea o modelo\n◉t de una muestra : Comparar una muestra con un valor esperado\n◉F-test o Levene: Comparar varianzas"
     elif 0.05 < p <= 0.10:
         return "Podría ser normal, pero con dudas"
     elif  0.01 < p <=0.05:
         return "Considera métodos no paramétricos"
     elif p <= 0.01:
         return "Usa análisis no paramétricos, o transforma los datos"


def medianaomedia(skew):

    if abs(skew) < 0.5:
     return "Distribución aproximadamente simétrica → usar media"
    else:
     return "Distribución sesgada → usar mediana"

def graficoapiladoconporcentaje(df, sns, plt, titulo, x, hue, tam_texto=8):
    ax = sns.countplot(data=df, x=x, hue=hue)
    total = len(df)
    for p in ax.patches:
        count = p.get_height()
        porcentaje = 100 * count / total
        ax.annotate(f'{porcentaje:.1f}%',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='bottom',
                    fontsize=tam_texto)  # 👈 aquí el tamaño
    plt.title(titulo)
    return plt.show()


def cantidad_de_columnas(df):
    for col in df.select_dtypes(include=object).columns:
     print(f"{col}{df[col].unique()}")
    return

def mostrar_primeras_filas(df,n):
    print(df.head(n))
    print(".......................................................................")

def mostrar_descripcion(df):
    print(df.describe())
    print(".......................................................................")

def mostrar_informacion(df):
    print(df.info())
    print(".......................................................................")

def cantidad_de_nulos(df):
    print("La cantidad de nulos :",df.isnull().sum())


def promedio_de_categoria(s,d,f,df):
    print("promedio",s,df.groupby(d)[f].mean())


def desviacion_estandar(columna_numerica,df):
    print("La desviacion estandar de ",columna_numerica,"es",df[columna_numerica].std())
    print(".......................................................................")


def cantidad_de_elementos_por_columna(name_columna,encabezado,elemento,df):
    print("En la columna ",name_columna,"hay",df[encabezado].count(),elemento)



def machine_simple(variables_predictoras,variable_a_predecir):
    variables_predictoras_train, variables_predictoras_test, variable_a_predecir_train, variable_a_predecir_test = train_test_split(variables_predictoras, variable_a_predecir, test_size=0.2, random_state=42)
    modelo = LinearRegression()
    modelo.fit(variables_predictoras_train, variable_a_predecir_train)
    variable_a_predecir_pred = modelo.predict(variables_predictoras_test)

    return r2_score(variable_a_predecir_test, variable_a_predecir_pred),mean_absolute_error(variable_a_predecir_test, variable_a_predecir_pred)

def posicion_rank(nueva_columna,columna_grupo,columna_calculada,df):
    df[nueva_columna] = df.groupby(columna_grupo)[columna_calculada].rank(method="dense", ascending=False)
    return df[nueva_columna]

def ordanenar_culman_menor_mayor(columna_a_ordenar,df):
    columna_ordenada = df.sort_values(by=columna_a_ordenar, ascending=False)
    return columna_ordenada

def mostrar_solo_un_grupo(colum_group,elemento_columngroup,columnord):
    print(columnord[columnord[colum_group] == elemento_columngroup])

def grafico_plot_promedio(columna_grupo,valorobjetivo_o_columnacaclulada,kind,titulo,etiquetay,df,modulo):
    df.groupby(columna_grupo)[valorobjetivo_o_columnacaclulada].mean().plot(kind=kind, title=titulo, ylabel=etiquetay)
    ax = df.groupby(columna_grupo)[valorobjetivo_o_columnacaclulada].mean().plot(kind=kind, title=titulo, ylabel=etiquetay)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    modulo.tight_layout()
    print("Grafico de promedio")
    return modulo.show()

def grafico_linea_por_anio(df, columna_fecha, columna_objetivo, titulo, etiqueta_y, modulo):
    df["Año"] = df[columna_fecha].dt.year
    resumen = df.groupby("Año")[columna_objetivo].sum().reset_index()
    ax = modulo.plot(resumen["Año"], resumen[columna_objetivo], marker="o")
    modulo.title(titulo)
    modulo.xlabel("Año")
    modulo.ylabel(etiqueta_y)
    modulo.grid(True)
    modulo.tight_layout()
    return modulo.show()

def histograma_seoborn(df,varbiale_onejtivo_calculada,bins,kde,titlle,xlabel,ylabel,modulo,m_para_mostrar_etquetas):
    modulo.histplot(df[varbiale_onejtivo_calculada],bins=bins,kde=kde)
    m_para_mostrar_etquetas.title(titlle)
    m_para_mostrar_etquetas.xlabel(xlabel)
    m_para_mostrar_etquetas.ylabel(ylabel)
    print("Grafico de frecuencia ")
    return m_para_mostrar_etquetas.show()

def categorizar_columna(df,columna,columna_nueva_categorizada,elemento_1,elemento_2):
    df[columna_nueva_categorizada] = df[columna].map({elemento_1: 1, elemento_2: 0})
    x=df[columna_nueva_categorizada]
    return x

def matriz_correlacion(df, colum1, colum2, colum3, lista_anios=None, columna_fecha="Fecha"):
    """
    Imprime la matriz de correlación entre 3 columnas, opcionalmente filtrando por años.

    Parámetros:
    - df: DataFrame
    - colum1, colum2, colum3: columnas numéricas a evaluar
    - lista_anios: lista opcional de años (ej. [2023])
    - columna_fecha: columna con fechas (por defecto "Fecha")
    """
    import pandas as pd

    columnas_corr = [colum1, colum2, colum3]

    if lista_anios:
        df[columna_fecha] = pd.to_datetime(df[columna_fecha])
        df = df[df[columna_fecha].dt.year.isin(lista_anios)]

    print("Matriz Correlación:\n", df[columnas_corr].corr())


def mapa_correlacion(df,colum1,colum2,colum3,modulo1_labels,modulo2,cmap,titulo):
    columnas_corr = [colum1, colum2, colum3]
    modulo2.heatmap(df[columnas_corr].corr(), annot=True, cmap=cmap)
    modulo1_labels.title(titulo)
    return modulo1_labels.show()

def tipo_de_distribucion(modulo_origen, shapiro, df, valor_obejtivo, lista_anios=None, columna_fecha="Fecha"):
    """
    Evalúa la normalidad de una variable (con Shapiro-Wilk), con opción de filtrar por rango de años.

    Parámetros:
    - modulo_origen: módulo donde está definida compartamientodis
    - shapiro: función de test de Shapiro-Wilk
    - df: DataFrame con los datos
    - valor_obejtivo: columna numérica a evaluar
    - lista_anios: lista opcional de años a filtrar (ej. [2023])
    - columna_fecha: columna con fecha (por defecto 'Fecha')
    """

    import pandas as pd

    if lista_anios:
        df[columna_fecha] = pd.to_datetime(df[columna_fecha])
        df = df[df[columna_fecha].dt.year.isin(lista_anios)]

    stat, p = shapiro(df[valor_obejtivo])
    print(modulo_origen.compartamientodis(p))

    if p > 0.05:
        return print("Valor de p es:", p, "\n✅ Puede considerarse una distribución normal")
    else:
        return print("Valor de p es:", p, "\n❌ No es distribución normal\n⚠️ *No usar métodos paramétricos*")



def tipo_tendencia_central(df,valor_obejetivo,modulo_origen):
    skew = df[valor_obejetivo].skew()
    print(modulo_origen.medianaomedia(skew))
    if 0.5>= skew >=-0.5 :
        return print("ES SIMETRICO, por ende usar media")
    else:
        return print("sesgo a la izquierda, por ende usar mediana")

def tabla_de_frecuencia(modul1,modulo2_labels,df,barras_por_campo,barras_divididas_por_sexo,titulo):
    modul1.countplot(data=df, x=barras_por_campo, hue=barras_divididas_por_sexo)
    modulo2_labels.title(titulo)
    modulo2_labels.xticks(rotation=45)
    return modulo2_labels.show()

def grafico_violin(sns,modulo_labelplt,df,x,ynum,titulo):
    sns.violinplot(data=df, x=x, y=ynum)
    modulo_labelplt.title(titulo)
    return modulo_labelplt.show()

def graficos_de_relaciones_cruzadas(sns,plt,df,columna1,columna2,columna3,columna4):
    sns.pairplot(df[[columna1, columna2, columna3, columna4]])
    return plt.show()

def grafico_circular(df,plt,categoria_a_evaluar,titulo):
    df[categoria_a_evaluar].value_counts().plot.pie(autopct="%1.1f%%")
    plt.title(titulo)
    plt.ylabel("")
    return plt.show()

def grafico_arbol(df,categoria1,categoria2,categoria3,valor_objetivo):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    X = df[[categoria1,categoria2,categoria3]]
    y = df[valor_objetivo]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_arbol = DecisionTreeRegressor(max_depth=4)
    modelo_arbol.fit(X_train, y_train)
    y_pred = modelo_arbol.predict(X_test)

    r2_arbol = r2_score(y_test, y_pred)
    mae_arbol = mean_absolute_error(y_test, y_pred)

    print("Árbol de Decisión:")
    print(f"R²: {r2_arbol}")
    print(f"MAE: {mae_arbol}")
    return r2_arbol, mae_arbol

def grafico_de_dispersion(sns,df,plt,x,y,titulo,etiquetax,etiquetay):
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(titulo)
    plt.xlabel(etiquetax)
    plt.ylabel(etiquetay)
    return plt.show()

def modelo_clasificacion_binaria(df, columnas_predictoras, columna_objetivo):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix

    X = df[columnas_predictoras]
    y = df[columna_objetivo]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = DecisionTreeClassifier()
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, predicciones)
    matriz = confusion_matrix(y_test, predicciones)

    print("Exactitud del modelo:", accuracy)
    print("Matriz de confusión:\n", matriz)


    return modelo, X, y_test, predicciones


def visualizar_arbol_y_confusion(modelo, X, y_test, predicciones):
    from sklearn.tree import plot_tree
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    """
    Dibuja el árbol de decisión y muestra la matriz de confusión.

    Parámetros:
    - modelo: Árbol entrenado (DecisionTreeClassifier o Regressor)
    - X: DataFrame con las variables predictoras (para obtener los nombres de columnas)
    - y_test: Valores reales de prueba
    - predicciones: Valores predichos por el modelo
    """
    plt.figure(figsize=(12, 8))
    plot_tree(modelo, feature_names=X.columns, class_names=["Reprobado", "Aprobado"], filled=True)
    plt.title("Árbol de Decisión")
    plt.show()
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, predicciones))

def prueba_t(grupo1,grupo2):
    from scipy.stats import levene, ttest_ind
    stat_levene, p_levene = levene(grupo1, grupo2)
    print("Levene p-valor:", p_levene)
    if p_levene > 0.05:
        stat_t, p_t = ttest_ind(grupo1, grupo2, equal_var=True)  # Student
        return print("t-test p-valor:", p_t, "\nmetodo usado *Student* ")
    else:
        stat_t, p_t = ttest_ind(grupo1, grupo2, equal_var=False)  # Welch
        return print("t-test p-valor:", p_t,"\nmetodo usado *Welch*")

def anova(df,valor_objetivo,columna_de_grupos):
    from scipy.stats import f_oneway
    grupos_anova = [grupo[valor_objetivo] for nombre, grupo in df.groupby(columna_de_grupos)]
    stat_anova, p_anova = f_oneway(*grupos_anova)
    print("ANOVA p-valor:", p_anova)

def mediana_o_medias(df, columna_objetivo, agrupacion_temporal=None, columna_categoria=None):
    from scipy.stats import skew
    import pandas as pd

    df = df.copy()

    if columna_objetivo not in df.columns:
        print(f"❌ La columna '{columna_objetivo}' no existe.")
        return

    if "Fecha" not in df.columns:
        print("❌ El DataFrame debe tener una columna 'Fecha'.")
        return

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    # Construcción de agrupación temporal
    if agrupacion_temporal:
        if agrupacion_temporal == "día":
            df["Agrupador_Tiempo"] = df["Fecha"].dt.dayofyear
            label = "Día del Año"
        elif agrupacion_temporal == "semana":
            df["Agrupador_Tiempo"] = df["Fecha"].dt.isocalendar().week
            label = "Semana"
        elif agrupacion_temporal == "mes":
            df["Agrupador_Tiempo"] = df["Fecha"].dt.month
            label = "Mes"
        elif agrupacion_temporal == "año":
            df["Agrupador_Tiempo"] = df["Fecha"].dt.year
            label = "Año"
        else:
            print(f"❌ agrupacion_temporal '{agrupacion_temporal}' no reconocida.")
            return
    else:
        df["Agrupador_Tiempo"] = None
        label = "Global"

    # Agrupar según el caso
    if columna_categoria:
        grupo = [columna_categoria, "Agrupador_Tiempo"] if agrupacion_temporal else [columna_categoria]
        datos = df.groupby(grupo)[columna_objetivo].sum()
        texto = f"'{columna_objetivo}' agrupado por '{columna_categoria}'" + (f" y '{label}'" if agrupacion_temporal else "")
    elif agrupacion_temporal:
        datos = df.groupby("Agrupador_Tiempo")[columna_objetivo].sum()
        texto = f"'{columna_objetivo}' por {label}"
    else:
        datos = df[columna_objetivo].dropna()
        texto = f"'{columna_objetivo}' global"

    # Evaluación
    if len(datos) < 3:
        print("❌ No hay suficientes datos para evaluar.")
        return

    media = datos.mean()
    mediana = datos.median()
    sesgo = skew(datos)

    print(f"\n📊 Evaluación de {texto}:")
    print(f" - Media   : ${media:,.2f}")
    print(f" - Mediana : ${mediana:,.2f}")
    print(f" - Sesgo   : {sesgo:.4f}")
    print("✅ Distribución aproximadamente simétrica → usar MEDIA" if abs(sesgo) < 0.5 else "⚠️ Distribución sesgada → usar MEDIANA")













def grafico_estacionalidad_ventas(df, fecha_col, monto_col, titulo, plt_mod):
    import pandas as pd
    import matplotlib.dates as mdates
    df[fecha_col] = pd.to_datetime(df[fecha_col])

    df["Año"] = df[fecha_col].dt.year
    df["Mes"] = df[fecha_col].dt.month

    agrupado = df.groupby(["Año", "Mes"])[monto_col].sum().reset_index()

    agrupado["Periodo"] = pd.to_datetime(
        agrupado.rename(columns={"Año": "year", "Mes": "month"}).assign(day=1)[["year", "month", "day"]]
    )

    agrupado = agrupado.sort_values("Periodo")

    fig, ax = plt_mod.subplots(figsize=(12, 5))
    ax.plot(agrupado["Periodo"], agrupado[monto_col], marker='o')

    # ✅ Separación: etiquetas cada 2 meses
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))

    ax.set_title(titulo)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Ventas")
    ax.grid(True)
    fig.autofmt_xdate(rotation=45)
    plt_mod.tight_layout()
    plt_mod.show()


def grafico_estacionalidad_por_dia_semana(df, fecha_col, columna_objetivo, titulo, plt_mod):
    """
    Gráfico de barras de estacionalidad por día de la semana.

    Parámetros:
    - df: DataFrame
    - fecha_col: nombre de la columna con fechas
    - columna_objetivo: columna de ventas a analizar
    - titulo: título del gráfico
    - plt_mod: módulo matplotlib.pyplot
    """
    import pandas as pd
    import calendar

    # Asegurar formato de fecha
    df[fecha_col] = pd.to_datetime(df[fecha_col])

    # Extraer día de la semana (0=Lunes, 6=Domingo)
    df["Dia_Semana"] = df[fecha_col].dt.dayofweek

    # Agrupar por día de la semana
    resumen = df.groupby("Dia_Semana")[columna_objetivo].sum().reset_index()

    # Mapear nombres de días
    resumen["Nombre_Dia"] = resumen["Dia_Semana"].map(dict(enumerate(calendar.day_name)))

    # Ordenar cronológicamente
    resumen = resumen.sort_values("Dia_Semana")

    # Graficar
    plt_mod.figure(figsize=(10, 5))
    plt_mod.bar(resumen["Nombre_Dia"], resumen[columna_objetivo], color="skyblue", edgecolor="black")
    plt_mod.title(titulo)
    plt_mod.xlabel("Día de la semana")
    plt_mod.ylabel("Total de " + columna_objetivo)
    plt_mod.xticks(rotation=45)
    plt_mod.grid(axis='y')
    plt_mod.tight_layout()
    plt_mod.show()

def grafico_estacionalidad_semanal_por_año(df, fecha_col, columna_objetivo, agno_deseado, titulo, plt_mod):
    import pandas as pd

    # Asegurar tipo datetime
    df[fecha_col] = pd.to_datetime(df[fecha_col])

    # Extraer semana y año
    df["Semana"] = df[fecha_col].dt.isocalendar().week
    df["Año"] = df[fecha_col].dt.year

    # Filtrar solo el año seleccionado
    df_filtrado = df[df["Año"] == agno_deseado]

    # Agrupar por semana
    agrupado = df_filtrado.groupby("Semana")[columna_objetivo].sum().reset_index()

    todas_semanas = pd.Series(range(1, 53), name="Semana")
    agrupado = agrupado.reindex(todas_semanas, fill_value=0)

    # Gráfico de barras
    plt_mod.figure(figsize=(12, 5))
    plt_mod.bar(agrupado["Semana"], agrupado[columna_objetivo])
    plt_mod.title(titulo)
    plt_mod.xlabel("Semana del Año")
    plt_mod.ylabel("Total de Ventas")
    plt_mod.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt_mod.tight_layout()
    plt_mod.show()




def grafico_estacionalidad_diaria_por_anio_rango(df, columna_fecha, columna_valor, lista_agno, titulo, plt):
    """
    Genera un gráfico de línea del promedio diario de una métrica para distintos años.

    Parámetros:
    - df: DataFrame con datos
    - columna_fecha: columna con fechas (debe ser datetime o convertible)
    - columna_valor: métrica a analizar (ej: Total_Compra)
    - lista_agno: lista de años a comparar (ej: [2023, 2024])
    - titulo: título del gráfico
    - plt: módulo matplotlib.pyplot
    """
    import pandas as pd

    df = df.copy()
    df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors='coerce')

    # Extraer año y día del año
    df["Año"] = df[columna_fecha].dt.year
    df["Día_Año"] = df[columna_fecha].dt.dayofyear

    # Filtrar años válidos
    df_filtrado = df[df["Año"].isin(lista_agno)]

    # Agrupar por día y año
    promedio = df_filtrado.groupby(["Año", "Día_Año"])[columna_valor].mean().reset_index()

    # Graficar
    plt.figure(figsize=(12, 6))
    for año in lista_agno:
        subset = promedio[promedio["Año"] == año]
        plt.plot(subset["Día_Año"], subset[columna_valor], label=str(año), linewidth=1.8)

    plt.title(titulo)
    plt.xlabel("Día del Año")
    plt.ylabel(f"Promedio de {columna_valor}")
    plt.grid(True)
    plt.legend(title="Año")
    plt.tight_layout()
    plt.show()

def comparar_estacionalidad_anual(df, columna_fecha, columna_objetivo, lista_anios, titulo, modulo):
    import pandas as pd
    import matplotlib.ticker as ticker

    # Asegurar formato datetime
    df[columna_fecha] = pd.to_datetime(df[columna_fecha])

    # Extraer año y día del año
    df["Año"] = df[columna_fecha].dt.year
    df["Día_del_Año"] = df[columna_fecha].dt.dayofyear

    # Filtrar los años deseados
    df_filtrado = df[df["Año"].isin(lista_anios)]

    # Agrupar por día del año y año
    agrupado = df_filtrado.groupby(["Día_del_Año", "Año"])[columna_objetivo].sum().reset_index()

    # Crear gráfico
    modulo.figure(figsize=(12, 6))

    for anio in lista_anios:
        datos_anio = agrupado[agrupado["Año"] == anio]
        modulo.plot(datos_anio["Día_del_Año"], datos_anio[columna_objetivo], label=f"Año {anio}")

    # Personalización del gráfico
    modulo.title(titulo, fontsize=14, weight='bold')
    modulo.xlabel("Día del Año")
    modulo.ylabel("Total de Ventas")
    modulo.legend()
    modulo.grid(True)
    modulo.tight_layout()

    # Formato de eje Y como dólares
    formato_dolares = ticker.FuncFormatter(lambda x, pos: f'${x:,.0f}')
    modulo.gca().yaxis.set_major_formatter(formato_dolares)

    modulo.show()



def comparar_estacionalidad_diaria_por_anios(df, columna_fecha, columna_objetivo, lista_anios, titulo, modulo):
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        """
        Genera un gráfico profesional que compara la estacionalidad diaria de las ventas entre varios años.

        Parámetros:
        - df: DataFrame con los datos.
        - columna_fecha: nombre de la columna con la fecha.
        - columna_objetivo: nombre de la columna con el valor numérico a analizar (ej. Total_Compra).
        - lista_anios: lista de años a comparar (ej. [2022, 2023]).
        - plt_mod: módulo matplotlib.pyplot (generalmente llamado plt).
        """

        # Asegurar que la columna de fecha esté en formato datetime
        df[columna_fecha] = pd.to_datetime(df[columna_fecha])

        # Crear columna auxiliar con el año y día del año
        df["Año"] = df[columna_fecha].dt.year
        df["Día_Año"] = df[columna_fecha].dt.dayofyear

        # Configurar gráfico
        modulo.figure(figsize=(14, 6))

        for i, anio in enumerate(lista_anios):
            df_anio = df[df["Año"] == anio]
            agrupado = df_anio.groupby("Día_Año")[columna_objetivo].sum().reset_index()
            modulo.plot(agrupado["Día_Año"], agrupado[columna_objetivo], label=f"Año {anio}")

        # Formateo profesional del eje Y en dólares
        modulo.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # Título y etiquetas
        años_str = " y ".join(str(a) for a in lista_anios)
        modulo.title(f"📈 Comparación Estacional de Ventas Diarias entre {años_str}", fontsize=14, fontweight='bold')
        modulo.xlabel("Día del Año")
        modulo.ylabel("Total de Ventas")
        modulo.grid(True)
        modulo.legend()
        modulo.tight_layout()
        modulo.show()






#uso definitivo
def grafico_comparacion_estacional_dias(
    df,
    columna_fecha,
    columna_objetivo,
    años_a_comparar,
    titulo,
    eventos=None,
    mostrar_texto_outliers=False,
    guardar_png=False,
    nombre_archivo=None,
    abreviar_valores=True
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import zscore

    def abreviar(valor):
        if valor >= 1_000_000:
            return f"${valor / 1_000_000:.2f}M"
        elif valor >= 1_000:
            return f"${valor / 1_000:.2f}K"
        else:
            return f"${valor:.2f}"

    df = df.copy()
    df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors='coerce')
    df["Año"] = df[columna_fecha].dt.year
    df["Día del Año"] = df[columna_fecha].dt.dayofyear
    df = df[df["Año"].isin(años_a_comparar)]

    df_prom = df.groupby(["Año", "Día del Año"])[columna_objetivo].mean().reset_index()

    # Detectar outliers
    pivot = df_prom.pivot(index="Día del Año", columns="Año", values=columna_objetivo)
    pivot["Promedio"] = pivot.mean(axis=1)
    pivot["Z"] = zscore(pivot["Promedio"])
    pivot["Es_Outlier"] = pivot["Z"].abs() > 2
    pivot = pivot.reset_index()
    df_outliers = pivot[pivot["Es_Outlier"]][["Día del Año", "Promedio", "Z"]]

    # Crear figura
    plt.figure(figsize=(14, 6))
    for año in años_a_comparar:
        datos = df_prom[df_prom["Año"] == año]
        plt.plot(datos["Día del Año"], datos[columna_objetivo], label=str(año), linewidth=1.2)

    # Dibujar eventos únicos (una sola vez por nombre)
    if eventos is not None and not eventos.empty:
        eventos = eventos.copy()
        eventos["Fecha"] = pd.to_datetime(eventos["Fecha"], dayfirst=True, errors='coerce')
        eventos["Día del Año"] = eventos["Fecha"].dt.dayofyear
        eventos_unicos = eventos.drop_duplicates(subset=["Evento"])

        for _, row in eventos_unicos.iterrows():
            dia = row["Día del Año"]
            plt.axvline(x=dia, color="gray", linestyle="--", alpha=0.7)
            plt.text(
                dia,
                plt.ylim()[1] * 0.98,
                row["Evento"],
                rotation=90,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='center',
                color='dimgray',
                fontweight='bold'
            )

    # Dibujar outliers
    if mostrar_texto_outliers:
        for _, row in df_outliers.iterrows():
            x = row["Día del Año"]
            y = row["Promedio"]
            z = row["Z"]

            # Pelota roja intensa
            plt.plot(x, y, 'o', color='darkred', markersize=6)

            # Posicionar texto
            offset = 100 if z > 0 else -120
            texto = abreviar(y) if abreviar_valores else f"${y:,.2f}"

            plt.text(
                x,
                y + offset,
                texto,
                color="black",
                fontsize=9,
                fontweight='bold',
                ha='center',
                va='bottom' if z > 0 else 'top'
            )

    plt.title(titulo)
    plt.xlabel("Día del Año")
    plt.ylabel(columna_objetivo)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Año")
    plt.tight_layout()

    if guardar_png:
        nombre_archivo = nombre_archivo or f"{titulo}.png"
        plt.savefig(nombre_archivo, dpi=300)

    plt.show()

    # Agregar columna de fecha a los outliers
    año_base = min(años_a_comparar)
    df_outliers["Fecha"] = pd.to_datetime(f"{año_base}-01-01") + pd.to_timedelta(df_outliers["Día del Año"] - 1, unit="D")
    df_outliers["Tipo"] = np.where(df_outliers["Z"] > 0, "↑ alto", "↓ bajo")

    return df_outliers









def histograma_seoborn_id_repetidos(
    df,
    columna_id_cliente,
    columna_id_orden,
    bins,
    kde,
    titulo,
    xlabel,
    ylabel,
    modulo,
    m_para_mostrar_etquetas,
    lista_anios=None,
    columna_fecha="Fecha"
):
    import pandas as pd

    if lista_anios:
        df[columna_fecha] = pd.to_datetime(df[columna_fecha])
        df = df[df[columna_fecha].dt.year.isin(lista_anios)]

    frecuencia = df.groupby(columna_id_cliente)[columna_id_orden].nunique()

    modulo.histplot(frecuencia, bins=bins, kde=kde)
    m_para_mostrar_etquetas.title(titulo)
    m_para_mostrar_etquetas.xlabel(xlabel)
    m_para_mostrar_etquetas.ylabel(ylabel)
    m_para_mostrar_etquetas.grid(True)
    m_para_mostrar_etquetas.tight_layout()
    print("✅ Histograma generado correctamente")
    return m_para_mostrar_etquetas.show()






def tipo_de_distribucion_id_repetidos(modulo_origen, shapiro, df, valor_obejtivo, frecuencia=False, col_cliente=None, col_orden=None, lista_anios=None, columna_fecha="Fecha"):
    import pandas as pd

    if lista_anios:
        df[columna_fecha] = pd.to_datetime(df[columna_fecha])
        df = df[df[columna_fecha].dt.year.isin(lista_anios)]

    if frecuencia:
        if col_cliente is None or col_orden is None:
            raise ValueError("Si frecuencia=True, debes proporcionar col_cliente y col_orden.")
        serie = df.groupby(col_cliente)[col_orden].nunique()
    else:
        serie = df[valor_obejtivo]

    stat, p = shapiro(serie)
    print(modulo_origen.compartamientodis(p))

    if p > 0.05:
        print("Valor de p:", p, "\n✅ Puede considerarse una distribución normal")
    else:
        print("Valor de p:", p, "\n❌ No es distribución normal\n⚠️ *No usar métodos paramétricos*")


def categorizar_columnas(df,columna,columna_nueva_categorizada,elemento_1,elemento_2,elemento_3,elemento_4):
    df[columna_nueva_categorizada] = df[columna].map({elemento_1: 1, elemento_2: 0,elemento_3:2,elemento_4:3})
    x=df[columna_nueva_categorizada]
    return x

def mapa_correlaciomapa_correlacion_por_agnio(df, colum1, colum2, colum3, modulo1_labels, modulo2, cmap, titulo, lista_anios=None, columna_fecha="Fecha"):
    """
    Mapa de calor de correlación para 3 columnas, con soporte de filtrado por lista de años.

    Parámetros:
    - df: DataFrame
    - colum1, colum2, colum3: columnas numéricas
    - modulo1_labels: módulo tipo matplotlib.pyplot
    - modulo2: módulo tipo seaborn
    - cmap: mapa de colores (ej: 'coolwarm')
    - titulo: título del gráfico
    - lista_anios: lista de años para filtrar (ej: [2023, 2024])
    - columna_fecha: nombre de la columna de fecha
    """
    import pandas as pd

    columnas_corr = [colum1, colum2, colum3]

    if lista_anios:
        df = df.copy()
        df[columna_fecha] = pd.to_datetime(df[columna_fecha])
        df = df[df[columna_fecha].dt.year.isin(lista_anios)]

    fig, ax = modulo1_labels.subplots(figsize=(8, 6))
    matriz = df[columnas_corr].corr()

    modulo2.heatmap(matriz, annot=True, cmap=cmap, fmt=".4f", linewidths=0.5, square=True, ax=ax)

    # Título
    if lista_anios:
        titulo += f" ({min(lista_anios)} - {max(lista_anios)})"
    ax.set_title(titulo, fontsize=14, pad=15)

    # Ajustes de etiquetas
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    modulo1_labels.tight_layout()
    return modulo1_labels.show()


def grafico_plot_total(columna_x, columna_y, tipo, titulo, etiqueta_y, df, plt, lista_anios=None, columna_fecha="Fecha"):
    """
    Genera un gráfico de barras con totales por categoría y muestra los valores encima de cada barra.

    Parámetros:
    - columna_x: categoría a agrupar
    - columna_y: valor numérico total a sumar
    - tipo: 'bar' o 'barh'
    - titulo: título del gráfico
    - etiqueta_y: etiqueta del eje Y
    - df: DataFrame
    - plt: módulo de matplotlib.pyplot
    - lista_anios: lista opcional de años para filtrar (ej: [2023, 2024])
    - columna_fecha: nombre de la columna de fecha (default = "Fecha")
    """
    import pandas as pd

    if lista_anios:
        df[columna_fecha] = pd.to_datetime(df[columna_fecha])
        df = df[df[columna_fecha].dt.year.isin(lista_anios)]

    df_total = df.groupby(columna_x)[columna_y].sum().reset_index()
    plt.figure(figsize=(8, 5))

    if tipo == "bar":
        barras = plt.bar(df_total[columna_x], df_total[columna_y], color="skyblue")
        plt.ylabel(etiqueta_y)
        plt.xlabel(columna_x)
        for barra in barras:
            altura = barra.get_height()
            plt.text(barra.get_x() + barra.get_width() / 2, altura, f'${altura:,.0f}',
                     ha='center', va='bottom', fontsize=9)

    elif tipo == "barh":
        barras = plt.barh(df_total[columna_x], df_total[columna_y], color="skyblue")
        plt.xlabel(etiqueta_y)
        plt.ylabel(columna_x)
        for barra in barras:
            ancho = barra.get_width()
            plt.text(ancho, barra.get_y() + barra.get_height() / 2, f'${ancho:,.0f}',
                     va='center', ha='left', fontsize=9)

    # Agregar rango de años al título si se usó
    if lista_anios:
        titulo += f" ({min(lista_anios)} - {max(lista_anios)})"

    plt.title(titulo)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def grafico_plot_promedio_con_valores(columna_grupo, valorobjetivo_o_columnacaclulada, kind, titulo, etiquetay, df, modulo):
    resumen = df.groupby(columna_grupo)[valorobjetivo_o_columnacaclulada].mean().reset_index()

    ax = resumen.plot(
        x=columna_grupo,
        y=valorobjetivo_o_columnacaclulada,
        kind=kind,
        title=titulo,
        ylabel=etiquetay,
        legend=False,
        color="skyblue"
    )

    # 🔁 Mostrar etiquetas sobre cada barra
    for i, v in enumerate(resumen[valorobjetivo_o_columnacaclulada]):
        if kind == "bar":
            ax.text(i, v, f"${v:,.0f}", ha='center', va='bottom', fontsize=9)
        elif kind == "barh":
            ax.text(v, i, f"${v:,.0f}", va='center', fontsize=9)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    modulo.tight_layout()
    print("Grafico de promedio")
    return modulo.show()








def grafico_estacionalidad_ventas_mes_por_agno_rango(df, columna_fecha, columna_objetivo, titulo, modulo, lista_anios=None):
    """
    Gráfico de barras de estacionalidad mensual, opcionalmente filtrando por rango de años.

    Parámetros:
    - df: DataFrame con los datos
    - columna_fecha: Nombre de la columna de fechas
    - columna_objetivo: Columna con el valor numérico a analizar
    - titulo: Título del gráfico
    - modulo: Módulo de visualización (e.g., plt)
    - lista_anios: Lista opcional de años a considerar (e.g., [2023, 2024])
    """
    import pandas as pd
    import calendar

    df[columna_fecha] = pd.to_datetime(df[columna_fecha])
    df["Año"] = df[columna_fecha].dt.year
    df["Mes"] = df[columna_fecha].dt.month

    if lista_anios:
        df = df[df["Año"].isin(lista_anios)]

    resumen = df.groupby("Mes")[columna_objetivo].sum().reset_index()
    resumen["Nombre_Mes"] = resumen["Mes"].apply(lambda x: calendar.month_name[x])
    resumen = resumen.sort_values("Mes")

    modulo.figure(figsize=(12, 5))
    barras = modulo.bar(resumen["Nombre_Mes"], resumen[columna_objetivo], color="lightblue", edgecolor="black")

    for barra in barras:
        altura = barra.get_height()
        modulo.text(barra.get_x() + barra.get_width() / 2, altura, f"${altura:,.0f}",
                    ha='center', va='bottom', fontsize=9)

    if lista_anios:
        titulo += f" ({min(lista_anios)} - {max(lista_anios)})"

    modulo.title(titulo)
    modulo.xlabel("Mes")
    modulo.ylabel("Total de " + columna_objetivo)
    modulo.xticks(rotation=45)
    modulo.grid(axis='y')
    modulo.tight_layout()
    modulo.show()


def grafico_plot_promedio_con_valores_por_rango(columna_categoria, columna_valor, tipo_grafico, titulo, etiqueta_y, df, modulo, anios=None, columna_fecha="Fecha"):
    """
    Gráfico con promedio de valores por categoría y etiquetas en cada barra.
    Parámetros:
    - columna_categoria: columna categórica a agrupar
    - columna_valor: columna numérica a promediar
    - tipo_grafico: 'bar' o 'barh'
    - titulo: título del gráfico
    - etiqueta_y: etiqueta del eje Y
    - df: DataFrame
    - modulo: módulo matplotlib.pyplot
    - anios: lista opcional de años para filtrar
    - columna_fecha: nombre de la columna de fecha
    """
    import pandas as pd

    if anios:
        df[columna_fecha] = pd.to_datetime(df[columna_fecha])
        df = df[df[columna_fecha].dt.year.isin(anios)]

    resumen = df.groupby(columna_categoria)[columna_valor].mean().sort_values(ascending=False)

    modulo.figure(figsize=(10, 5))
    barras = resumen.plot(kind=tipo_grafico, color='skyblue', edgecolor='black')

    # Agregar etiquetas
    for i, valor in enumerate(resumen):
        if tipo_grafico == "bar":
            barras.text(i, valor, f"${valor:,.0f}", ha='center', va='bottom', fontsize=9)
        else:
            barras.text(valor, i, f"${valor:,.0f}", va='center', fontsize=9)

    if anios:
        titulo += f" ({min(anios)} - {max(anios)})"

    modulo.title(titulo)
    modulo.ylabel(etiqueta_y)
    modulo.xlabel(columna_categoria)
    modulo.xticks(rotation=45)
    modulo.grid(axis='y')
    modulo.tight_layout()
    modulo.show()


def grafico_plot_promedio_con_valores_por_rango_tiempo(columna_grupo, columna_valor, tipo_grafico, titulo, etiqueta_y, df, modulo, lista_anios, columna_fecha):
    import pandas as pd

    df[columna_fecha] = pd.to_datetime(df[columna_fecha])
    df_filtrado = df[df[columna_fecha].dt.year.isin(lista_anios)]

    resumen = df_filtrado.groupby(columna_grupo)[columna_valor].mean().sort_values(ascending=False)

    modulo.figure(figsize=(10, 5))
    ax = resumen.plot(kind=tipo_grafico, color='skyblue', edgecolor='black')

    for i, valor in enumerate(resumen):
        if tipo_grafico == "bar":
            ax.text(i, valor, f"${valor:,.0f}", ha='center', va='bottom', fontsize=9)
        else:
            ax.text(valor, i, f"${valor:,.0f}", va='center', fontsize=9)

    modulo.title(titulo)
    modulo.ylabel(etiqueta_y)
    modulo.xlabel(columna_grupo)
    modulo.xticks(rotation=45)
    modulo.grid(axis='y')
    modulo.tight_layout()
    modulo.show()

def grafico_circular_por_rango_tiempo(df, plt, categoria_a_evaluar, titulo, lista_anios, columna_fecha):
    import pandas as pd
    df[columna_fecha] = pd.to_datetime(df[columna_fecha])
    df = df[df[columna_fecha].dt.year.isin(lista_anios)]

    df[categoria_a_evaluar].value_counts().plot.pie(autopct="%1.1f%%")
    plt.title(titulo)
    plt.ylabel("")
    plt.tight_layout()
    return plt.show()


def graficoapiladoconporcentaje_por_agno(df, sns, plt, titulo, x, hue, tam_texto, lista_anios, columna_fecha):
    import pandas as pd
    df[columna_fecha] = pd.to_datetime(df[columna_fecha])
    df = df[df[columna_fecha].dt.year.isin(lista_anios)]

    ax = sns.countplot(data=df, x=x, hue=hue)
    total = len(df)
    for p in ax.patches:
        count = p.get_height()
        porcentaje = 100 * count / total
        ax.annotate(f'{porcentaje:.1f}%',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='bottom',
                    fontsize=tam_texto)
    plt.title(titulo)
    plt.tight_layout()
    return plt.show()


def modelo_random_forest_total_compra(df, columnas_entrada, columna_objetivo, años, fecha_col):
    """
    Entrena un modelo Random Forest para predecir columna_objetivo.

    Retorna (igual que antes):
    - modelo entrenado
    - MAE
    - R2
    - dict con:
        { "feature_importance": {...},
          "metrics": {"WAPE": float, "MASE": float|nan, "sMAPE": float, "Bias": float, "Bias_pct": float} }
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # Asegurar tipo datetime para filtrar por año y ordenar para MASE
    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce")

    # Filtrar por años solicitados
    df_filtrado = df[df[fecha_col].dt.year.isin(años)].copy()

    # Separar variables
    X = df_filtrado[columnas_entrada]
    y = df_filtrado[columna_objetivo]

    # Split (conserva índices para vincular con fechas en MASE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modelo
    modelo = RandomForestRegressor(random_state=42)
    modelo.fit(X_train, y_train)

    # Predicción y evaluación base
    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # ================== MÉTRICAS ADICIONALES ==================
    eps = 1e-8
    y_true = y_test.values.astype(float)
    y_hat  = y_pred.astype(float)

    # WAPE = sum(|e|)/sum(|y|)
    wape = float(np.sum(np.abs(y_true - y_hat)) / (np.sum(np.abs(y_true)) + eps))

    # sMAPE = mean( 2|e| / (|y|+|y_hat|) )
    smape = float(np.mean(2.0 * np.abs(y_true - y_hat) / (np.abs(y_true) + np.abs(y_hat) + eps)))

    # Bias (ME) y Bias porcentual (tipo PBIAS con denominador |y|)
    bias = float(np.mean(y_hat - y_true))
    bias_pct = float((np.sum(y_hat - y_true)) / (np.sum(np.abs(y_true)) + eps))

    # MASE: MAE / MAE_naive; naive no estacional (m=1) en el CONJUNTO DE ENTRENAMIENTO
    # Se ordena por fecha para calcular diferencias consecutivas significativas
    try:
        df_train = df_filtrado.loc[y_train.index, [fecha_col, columna_objetivo]].copy()
        df_train = df_train.sort_values(fecha_col)
        naive_errors = np.abs(
            df_train[columna_objetivo].values[1:] - df_train[columna_objetivo].values[:-1]
        )
        denom = np.mean(naive_errors) if len(naive_errors) > 0 else np.nan
        mase = float(mae / (denom + eps)) if denom is not np.nan else float("nan")
    except Exception:
        mase = float("nan")  # fallback en caso de algún problema con el ordenamiento

    metrics_ext = {
        "WAPE": wape,          # fracción (0–1)
        "MASE": mase,          # puede ser NaN si no hay suficientes puntos
        "sMAPE": smape,        # fracción (0–1)
        "Bias": bias,          # misma unidad que la variable objetivo
        "Bias_pct": bias_pct   # fracción (0–1), tipo PBIAS con denominador |y|
    }
    # ==========================================================

    # Importancias
    importancias = modelo.feature_importances_
    importancia_dict = dict(zip(columnas_entrada, importancias))

    # Gráfico con valores en barras
    plt.figure(figsize=(10, 5))
    bars = plt.barh(columnas_entrada, importancias, color="steelblue", edgecolor="black")
    for bar, imp in zip(bars, importancias):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{imp:.2f}", va='center', fontsize=10)
    plt.title("Importancia de variables")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.show()

    # Métricas en consola
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")
    print(f"WAPE: {wape*100:.2f}% | sMAPE: {smape*100:.2f}% | MASE: {mase:.3f} | Bias: {bias:.2f} ({bias_pct*100:.2f}%)")

    # Mantener la compatibilidad del retorno (4 elementos)
    extra = {"feature_importance": importancia_dict, "metrics": metrics_ext}
    return modelo, mae, r2, extra



def categorizar_columnas_con_lista_agnos(df, columna_original, nueva_columna, lista_categorias):
    """
    Crea una nueva columna categorizada respetando un orden específico de categorías.

    Parámetros:
    - df: DataFrame con los datos
    - columna_original: columna a convertir (categórica)
    - nueva_columna: nombre de la nueva columna con valores numéricos
    - lista_categorias: lista que define el orden exacto de las categorías, ej: ["Oficina", "Moda", "Hogar"]

    Retorna:
    - df con nueva columna codificada
    - diccionario de mapeo
    """
    # Crear diccionario de mapeo respetando el orden de la lista
    mapa = {cat: i for i, cat in enumerate(lista_categorias)}

    # Aplicar mapeo
    df[nueva_columna] = df[columna_original].map(mapa)

    return df, mapa


def predecir_datos_avanzado(df, modelo, columnas_entrada, anios, fecha_col, nombre_columna_prediccion="Prediccion"):
    """
    Realiza predicciones sobre un DataFrame usando un modelo ya entrenado.

    Parámetros:
    - df: DataFrame original con los datos.
    - modelo: modelo ya entrenado (por ejemplo, RandomForestRegressor).
    - columnas_entrada: lista de columnas que se usarán como entrada para el modelo.
    - anios: lista de años a filtrar, ej: [2023, 2024].
    - fecha_col: nombre de la columna de fechas.
    - nombre_columna_prediccion: nombre de la columna donde se guardará la predicción.

    Retorna:
    - df_filtrado: DataFrame con columna de predicción añadida.
    """
    import pandas as pd

    # Filtrar por los años indicados
    df_filtrado = df[df[fecha_col].dt.year.isin(anios)].copy()

    # Validar que las columnas de entrada existen
    for col in columnas_entrada:
        if col not in df_filtrado.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame.")

    # Aplicar el modelo para predecir
    df_filtrado[nombre_columna_prediccion] = modelo.predict(df_filtrado[columnas_entrada])

    return df_filtrado


def grafico_predicciones_barras_agrupadas(df, x_col, hue_col, y_col, titulo, ylabel, plt_mod, sns_mod):
    """
    Gráfico de barras agrupadas para mostrar predicciones por categorías múltiples.

    Parámetros:
    - df: DataFrame con columnas categóricas y predicción
    - x_col: columna categórica para eje X (ej. 'Categoría')
    - hue_col: columna para agrupación de color (ej. 'Método_Pago')
    - y_col: columna con valores predichos (ej. 'Prediccion_Ingreso')
    - titulo: título del gráfico
    - ylabel: etiqueta del eje Y
    - plt_mod: módulo matplotlib.pyplot
    - sns_mod: módulo seaborn
    """
    plt_mod.figure(figsize=(12, 6))
    ax = sns_mod.barplot(data=df, x=x_col, y=y_col, hue=hue_col)

    for container in ax.containers:
        ax.bar_label(container, fmt='${:.0f}', label_type='edge', fontsize=8)

    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_col)
    plt_mod.xticks(rotation=45)
    plt_mod.tight_layout()
    plt_mod.show()

def grafico_comparativo_real_vs_predicho(df, categoria_col, metodo_pago_col, ingreso_real_col, ingreso_predicho_col, titulo, etiqueta_eje_y, plt, sns):
    """
    Muestra un gráfico de barras comparando ingreso real vs predicho por categoría y método de pago.

    Parámetros:
    - df: DataFrame con columnas de categoría, método de pago, ingreso real y predicho.
    - categoria_col: columna con las categorías de producto.
    - metodo_pago_col: columna con los métodos de pago.
    - ingreso_real_col: columna con los ingresos reales (ej. 'Ingreso_Item').
    - ingreso_predicho_col: columna con las predicciones del modelo.
    - titulo: título del gráfico.
    - etiqueta_eje_y: etiqueta del eje Y.
    - plt, sns: librerías para graficar.
    """

    df_agrupado = df.groupby([categoria_col, metodo_pago_col]).agg(
        Ingreso_Real=(ingreso_real_col, "mean"),
        Ingreso_Predicho=(ingreso_predicho_col, "mean")
    ).reset_index()

    df_melt = df_agrupado.melt(id_vars=[categoria_col, metodo_pago_col],
                               value_vars=["Ingreso_Real", "Ingreso_Predicho"],
                               var_name="Tipo",
                               value_name="Valor")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x=categoria_col, y="Valor", hue="Tipo", palette="Set2", ci=None)
    plt.title(titulo)
    plt.ylabel(etiqueta_eje_y)
    plt.xlabel(categoria_col)
    plt.legend(title="Ingreso")
    plt.tight_layout()
    plt.show()


def grafico_comparativo_real_vs_predicho_agrupado(
    df,
    categoria_col,
    metodo_pago_col,
    ingreso_real_col,
    ingreso_predicho_col,
    titulo,
    etiqueta_eje_y,
    plt,
    sns
):
    """
    Gráfico comparativo entre ingreso real y predicho, agrupado por categoría y método de pago,
    con etiquetas legibles y sin saturación.

    Parámetros:
    - df: DataFrame con los datos
    - categoria_col: columna con categorías (eje X)
    - metodo_pago_col: columna con métodos de pago
    - ingreso_real_col: columna con ingresos reales
    - ingreso_predicho_col: columna con ingresos predichos
    - titulo: título del gráfico
    - etiqueta_eje_y: texto para eje Y
    - plt, sns: librerías de visualización
    """
    import pandas as pd

    # Agrupar promedios
    df_agrupado = df.groupby([categoria_col, metodo_pago_col]).agg({
        ingreso_real_col: 'mean',
        ingreso_predicho_col: 'mean'
    }).reset_index()

    # Preparar formato largo
    df_real = df_agrupado[[categoria_col, metodo_pago_col, ingreso_real_col]].copy()
    df_real["Tipo"] = "Real"
    df_real = df_real.rename(columns={ingreso_real_col: "Ingreso"})

    df_pred = df_agrupado[[categoria_col, metodo_pago_col, ingreso_predicho_col]].copy()
    df_pred["Tipo"] = "Predicho"
    df_pred = df_pred.rename(columns={ingreso_predicho_col: "Ingreso"})

    df_melt = pd.concat([df_real, df_pred])
    df_melt["Grupo"] = df_melt[metodo_pago_col] + " - " + df_melt["Tipo"]

    # Gráfico
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(
        data=df_melt,
        x=categoria_col,
        y="Ingreso",
        hue="Grupo",
        ci=None
    )

    # Etiquetas más pequeñas, visibles y limpias
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='${:.0f}',
            padding=2,
            fontsize=8,
            label_type='edge'
        )

    plt.title(titulo, fontsize=13)
    plt.ylabel(etiqueta_eje_y)
    plt.xlabel(categoria_col)
    plt.xticks(rotation=30, ha='right')
    plt.legend(title="Método - Tipo", fontsize=9, title_fontsize=10)
    plt.tight_layout()
    plt.show()

def grafico_lineas_ingresos_por_anio(df, grupo_col, real_col, pred_col, titulo, anios, fecha_col, plt_mod):
    """
    Gráfico de líneas por año: compara ingresos reales vs predichos.

    Parámetros:
    - df: DataFrame
    - grupo_col: columna agrupadora (e.g., 'Categoría')
    - real_col: ingresos reales (e.g., 'Ingreso_Item')
    - pred_col: ingresos predichos (e.g., 'Prediccion_Ingreso')
    - titulo: título del gráfico
    - anios: lista de años a incluir [2023, 2024]
    - fecha_col: columna de fechas
    - plt_mod: módulo matplotlib.pyplot
    """
    import pandas as pd
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df_filtrado = df[df[fecha_col].dt.year.isin(anios)]

    resumen = df_filtrado.groupby(grupo_col)[[real_col, pred_col]].sum().reset_index()
    resumen = resumen.sort_values(grupo_col)

    plt_mod.figure(figsize=(10, 6))
    plt_mod.plot(resumen[grupo_col], resumen[real_col], marker='o', label='Real', color='skyblue')
    plt_mod.plot(resumen[grupo_col], resumen[pred_col], marker='o', label='Predicho', color='orange')

    for i in range(len(resumen)):
        plt_mod.text(resumen[grupo_col][i], resumen[real_col][i], f"${resumen[real_col][i]:,.0f}", ha='center', va='bottom', fontsize=8)
        plt_mod.text(resumen[grupo_col][i], resumen[pred_col][i], f"${resumen[pred_col][i]:,.0f}", ha='center', va='top', fontsize=8)

    plt_mod.title(titulo)
    plt_mod.xlabel(grupo_col)
    plt_mod.ylabel("Ingreso")
    plt_mod.xticks(rotation=45)
    plt_mod.legend()
    plt_mod.grid(True, linestyle='--', alpha=0.5)
    plt_mod.tight_layout()
    plt_mod.show()


def grafico_lollipop_ingresos_por_anio(df, grupo_col, real_col, pred_col, titulo, anios, fecha_col, plt_mod):
    """
    Gráfico tipo Lollipop por año: compara valores reales vs predichos.

    Parámetros:
    - df: DataFrame con los datos
    - grupo_col: columna categórica (e.g., 'Método_Pago', 'Categoría')
    - real_col: columna con valores reales (e.g., 'Ingreso_Item')
    - pred_col: columna con predicciones (e.g., 'Prediccion_Ingreso')
    - titulo: título del gráfico
    - anios: lista de años a incluir [2023, 2024]
    - fecha_col: columna con las fechas
    - plt_mod: módulo matplotlib.pyplot
    """
    import pandas as pd
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df_filtrado = df[df[fecha_col].dt.year.isin(anios)]

    resumen = df_filtrado.groupby(grupo_col)[[real_col, pred_col]].sum().reset_index()
    resumen = resumen.sort_values(real_col)

    plt_mod.figure(figsize=(10, 6))
    plt_mod.hlines(y=resumen[grupo_col], xmin=resumen[real_col], xmax=resumen[pred_col], color='gray', alpha=0.7)
    plt_mod.plot(resumen[real_col], resumen[grupo_col], 'o', label='Real', color='skyblue',markersize=10)
    plt_mod.plot(resumen[pred_col], resumen[grupo_col], 'o', label='Predicho', color='orange',markersize=10)

    for _, row in resumen.iterrows():
        plt_mod.text(row[real_col], row[grupo_col], f"${row[real_col]:,.0f}", va='center', ha='right', fontsize=8)
        plt_mod.text(row[pred_col], row[grupo_col], f"${row[pred_col]:,.0f}", va='center', ha='left', fontsize=8)

    plt_mod.title(titulo)
    plt_mod.xlabel("Ingreso")
    plt_mod.ylabel(grupo_col)
    plt_mod.legend()
    plt_mod.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt_mod.tight_layout()
    plt_mod.show()


def predecir_meses_faltantes_2025(df, modelo, columnas_entrada, fecha_col, variable_objetivo, nombre_columna_prediccion="Prediccion_Ingreso"):
    """
    Predice los valores de los meses faltantes en 2025 usando un modelo entrenado.

    Parámetros:
    - df: DataFrame completo con datos reales
    - modelo: modelo de ML ya entrenado
    - columnas_entrada: lista de columnas utilizadas como entrada al modelo
    - fecha_col: nombre de la columna con fechas
    - variable_objetivo: nombre de la columna objetivo (e.g., 'Ingreso_Item')
    - nombre_columna_prediccion: nombre de la columna donde guardar predicción

    Retorna:
    - df_combinado: DataFrame con datos reales y predichos para 2025
    """
    import pandas as pd

    # Asegurarse que fecha esté en datetime
    df[fecha_col] = pd.to_datetime(df[fecha_col])

    # Filtrar 2025 y separar reales
    df_2025 = df[df[fecha_col].dt.year == 2025].copy()
    df_2025["Mes"] = df_2025[fecha_col].dt.month

    # Obtener meses reales ya presentes
    meses_presentes = df_2025["Mes"].unique().tolist()
    todos_meses = list(range(1, 13))
    meses_faltantes = [m for m in todos_meses if m not in meses_presentes]

    # Generar datos sintéticos para los meses faltantes
    nuevas_filas = []
    for mes in meses_faltantes:
        # Asumimos una fecha media del mes
        fecha_sintetica = pd.Timestamp(year=2025, month=mes, day=15)

        # Crear una fila por cada combinación única relevante
        for _, fila_base in df[df[fecha_col].dt.year == 2024].drop_duplicates(subset=columnas_entrada).iterrows():
            nueva_fila = fila_base[columnas_entrada].copy()
            nueva_fila[fecha_col] = fecha_sintetica
            nuevas_filas.append(nueva_fila)

    df_pred = pd.DataFrame(nuevas_filas)

    # Predecir usando el modelo
    df_pred[nombre_columna_prediccion] = modelo.predict(df_pred[columnas_entrada])
    df_pred[fecha_col] = pd.to_datetime(df_pred[fecha_col])
    df_pred["Origen"] = "Predicho"

    # Agregar identificador
    df_pred["ID_Prediccion"] = ["PRED_" + str(i) for i in range(len(df_pred))]

    # Preparar datos reales
    df_reales = df_2025.copy()
    df_reales["Origen"] = "Real"
    df_reales[nombre_columna_prediccion] = df_reales[variable_objetivo]

    # Unir
    columnas_salida = list(set(df_reales.columns).intersection(df_pred.columns))
    df_combinado = pd.concat([df_reales[columnas_salida], df_pred[columnas_salida]], ignore_index=True)

    return df_combinado

def grafico_lineas_prediccion_vs_real_sns(df, fecha_col, real_col, pred_col, titulo, sns_mod, plt_mod):
    """
    Gráfico de líneas para comparar ingreso real vs predicho por fecha.
    Usa seaborn.lineplot.
    """
    import pandas as pd

    # Asegurar fechas
    df[fecha_col] = pd.to_datetime(df[fecha_col])

    # Agrupar
    resumen = df.groupby(fecha_col)[[real_col, pred_col]].sum().reset_index()

    # Convertir a formato largo
    df_melt = resumen.melt(id_vars=fecha_col, value_vars=[real_col, pred_col],
                           var_name="Tipo", value_name="Ingreso")

    # Reemplazar nombres para mayor claridad
    df_melt["Tipo"] = df_melt["Tipo"].replace({real_col: "Ingreso Real", pred_col: "Ingreso Predicho"})

    # Graficar
    plt_mod.figure(figsize=(12, 6))
    sns_mod.lineplot(data=df_melt, x=fecha_col, y="Ingreso", hue="Tipo", marker="o", errorbar=None)
    plt_mod.title(titulo)
    plt_mod.xlabel("Fecha")
    plt_mod.ylabel("Ingreso")
    plt_mod.grid(True, linestyle='--', alpha=0.5)
    plt_mod.tight_layout()
    plt_mod.xticks(rotation=45)
    plt_mod.legend()
    plt_mod.show()

def grafico_lineas_prediccion_vs_real_sns_con_anios(df, fecha_col, real_col, pred_col, titulo, anios, sns_mod, plt_mod):
    """
    Gráfico de líneas (seaborn) comparando valores reales y predichos por fecha, filtrado por años.

    Parámetros:
    - df: DataFrame
    - fecha_col: columna de fechas
    - real_col: columna con valores reales
    - pred_col: columna con valores predichos
    - titulo: título del gráfico
    - anios: lista de años a incluir [2023, 2024, 2025...]
    - sns_mod: módulo seaborn
    - plt_mod: módulo matplotlib.pyplot
    """
    import pandas as pd

    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df_filtrado = df[df[fecha_col].dt.year.isin(anios)].copy()

    resumen = df_filtrado.groupby(fecha_col)[[real_col, pred_col]].sum().reset_index()

    df_melt = resumen.melt(id_vars=fecha_col, value_vars=[real_col, pred_col],
                           var_name="Tipo", value_name="Ingreso")

    df_melt["Tipo"] = df_melt["Tipo"].replace({real_col: "Ingreso Real", pred_col: "Ingreso Predicho"})

    plt_mod.figure(figsize=(12, 6))
    sns_mod.lineplot(data=df_melt, x=fecha_col, y="Ingreso", hue="Tipo", marker="o", errorbar=None)
    plt_mod.title(titulo)
    plt_mod.xlabel("Fecha")
    plt_mod.ylabel("Ingreso")
    plt_mod.grid(True, linestyle='--', alpha=0.5)
    plt_mod.tight_layout()
    plt_mod.xticks(rotation=45)
    plt_mod.legend()
    plt_mod.show()


def grafico_lineas_prediccion_vs_real_sns_con_anios(df, fecha_col, real_col, pred_col, titulo, anios, sns_mod, plt_mod):
    """
    Gráfico de líneas con seaborn para comparar valores reales y predichos por fecha.

    Parámetros:
    - df: DataFrame con datos
    - fecha_col: columna de fechas
    - real_col: columna con valores reales (por ejemplo 'Ingreso_Item')
    - pred_col: columna con valores predichos (por ejemplo 'Prediccion_Ingreso')
    - titulo: título del gráfico
    - anios: lista de años que se desean graficar, ej. [2025]
    - sns_mod: módulo seaborn
    - plt_mod: módulo matplotlib.pyplot
    """

    import pandas as pd

    # Asegurar formato de fecha
    df[fecha_col] = pd.to_datetime(df[fecha_col])

    # Filtrar por los años deseados
    df_filtrado = df[df[fecha_col].dt.year.isin(anios)].copy()
    if df_filtrado.empty:
        print(f"No hay datos disponibles para los años: {anios}")
        return

    # Agrupar por fecha
    resumen = df_filtrado.groupby(fecha_col)[[real_col, pred_col]].sum().reset_index()

    # Transformar a formato largo
    df_melt = resumen.melt(
        id_vars=fecha_col,
        value_vars=[real_col, pred_col],
        var_name="Tipo",
        value_name="Ingreso"
    )

    # Renombrar para que la leyenda sea clara
    df_melt["Tipo"] = df_melt["Tipo"].replace({
        real_col: "Ingreso Real",
        pred_col: "Ingreso Predicho"
    })

    # Graficar
    plt_mod.figure(figsize=(12, 6))
    sns_mod.lineplot(data=df_melt, x=fecha_col, y="Ingreso", hue="Tipo", marker="o", errorbar=None)

    plt_mod.title(titulo)
    plt_mod.xlabel("Fecha")
    plt_mod.ylabel("Ingreso")
    plt_mod.xticks(rotation=45)
    plt_mod.grid(True, linestyle='--', alpha=0.5)
    plt_mod.legend()
    plt_mod.tight_layout()
    plt_mod.show()


def predecir_meses_faltantes_2025_F(df, modelo, columnas_entrada, fecha_col, nombre_columna_prediccion):
    import pandas as pd

    df_2025 = df.copy()
    df_2025["Año"] = pd.to_datetime(df_2025[fecha_col]).dt.year
    df_2025 = df_2025[df_2025["Año"] == 2025]

    # Separar reales y predichos según si la columna de predicción ya tiene valores
    df_real = df_2025[df_2025[nombre_columna_prediccion].notnull()].copy()
    df_real["Origen"] = "Real"

    df_pred = df_2025[df_2025[nombre_columna_prediccion].isnull()].copy()
    df_pred[nombre_columna_prediccion] = modelo.predict(df_pred[columnas_entrada])
    df_pred["Origen"] = "Predicho"

    # Concatenar todo
    df_completo = pd.concat([df_real, df_pred], ignore_index=True)

    # Agregar columna "Categoría" desde "Categoría_categorizada"
    categorias = ["Oficina", "Hogar", "Juguetería", "Moda", "Herramientas", "Electrónica"]
    mapa_categorias = {i: cat for i, cat in enumerate(categorias)}
    df_completo["Categoría"] = df_completo["Categoría_categorizada"].map(mapa_categorias)

    return df_completo




def grafico_lineas_ingresos_reales_vs_predichos_mensual(df, fecha_col, col_real, col_pred, titulo, anio, plt_mod, sns_mod):
    """
    Gráfico de líneas mensuales comparando ingreso real y predicho.
    """

    import pandas as pd

    # Asegurar fechas
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df_filtrado = df[df[fecha_col].dt.year == anio].copy()

    # Crear columna de Mes
    df_filtrado["Mes"] = df_filtrado[fecha_col].dt.to_period("M").dt.to_timestamp()

    # Crear columnas auxiliares para graficar
    df_real = df_filtrado[df_filtrado["Origen"] == "Real"].groupby("Mes")[col_real].sum().reset_index()
    df_real["Tipo"] = "Ingreso Real"
    df_real.rename(columns={col_real: "Ingreso"}, inplace=True)

    df_pred = df_filtrado[df_filtrado["Origen"] == "Predicho"].groupby("Mes")[col_pred].sum().reset_index()
    df_pred["Tipo"] = "Ingreso Predicho"
    df_pred.rename(columns={col_pred: "Ingreso"}, inplace=True)

    df_final = pd.concat([df_real, df_pred], ignore_index=True)

    # Plot
    plt_mod.figure(figsize=(12, 6))
    sns_mod.lineplot(data=df_final, x="Mes", y="Ingreso", hue="Tipo", marker="o")
    plt_mod.title(titulo)
    plt_mod.xlabel("Mes")
    plt_mod.ylabel("Ingreso Total")
    plt_mod.xticks(rotation=45)
    plt_mod.grid(True, linestyle='--', alpha=0.5)
    plt_mod.tight_layout()
    plt_mod.show()




def grafico_linea_real_vs_predicho_por_categoria_2025(df_2025, categoria_col, col_real, col_pred, fecha_col, plt, sns):
    """
    Grafica las líneas de ingreso real vs. predicho para el año 2025 por categoría.

    - df_2025: DataFrame con datos reales y predichos (con columna 'Origen')
    - categoria_col: columna de categoría
    - col_real: nombre de columna real (usualmente Ingreso_Item, si no hay, usa Prediccion_Ingreso)
    - col_pred: nombre de columna predicha
    - fecha_col: columna con la fecha
    - plt, sns: módulos importados
    """
    df_2025 = df_2025.copy()
    df_2025["Mes"] = df_2025[fecha_col].dt.month

    df_agrupado = df_2025.groupby(["Mes", categoria_col, "Origen"])[col_pred].sum().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_agrupado, x="Mes", y=col_pred, hue=categoria_col, style="Origen", markers=True)
    plt.title("Ingreso Real vs Predicho por Categoría - Año 2025")
    plt.xlabel("Mes")
    plt.ylabel("Ingreso")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def grafico_linea_real_vs_predicho_por_categoria_2025f(
    df, categoria_col, col_prediccion, titulo, col_eje_y, plt, sns
):
    import pandas as pd

    df_2025 = df.copy()
    df_2025["Año"] = pd.to_datetime(df_2025["Fecha"]).dt.year
    df_2025 = df_2025[df_2025["Año"] == 2025]

    # Agrupar por categoría y origen
    df_grouped = df_2025.groupby([categoria_col, "Origen"])[col_prediccion].sum().reset_index()

    # Separar por tipo de dato
    df_real = df_grouped[df_grouped["Origen"] == "Real"]
    df_pred = df_grouped[df_grouped["Origen"] == "Predicho"]

    # Ordenar las categorías por ingreso real
    categorias_ordenadas = df_real.sort_values(by=col_prediccion, ascending=False)[categoria_col]
    categorias_ordenadas = categorias_ordenadas.tolist()

    df_real[categoria_col] = pd.Categorical(df_real[categoria_col], categories=categorias_ordenadas, ordered=True)
    df_pred[categoria_col] = pd.Categorical(df_pred[categoria_col], categories=categorias_ordenadas, ordered=True)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_real[categoria_col], df_real[col_prediccion], marker='o', color='skyblue', label='Real')
    plt.plot(df_pred[categoria_col], df_pred[col_prediccion], marker='o', color='orange', label='Predicho')

    # Etiquetas
    for x, y in zip(df_real[categoria_col], df_real[col_prediccion]):
        plt.text(x, y, f"${y:,.0f}", ha='center', va='bottom', fontsize=9)

    for x, y in zip(df_pred[categoria_col], df_pred[col_prediccion]):
        plt.text(x, y, f"${y:,.0f}", ha='center', va='top', fontsize=9)

    plt.title(titulo)
    plt.xlabel("Categoría")
    plt.ylabel(col_eje_y)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.text(
        0.5, 0.05,
        "⚠️ Diferencia visual por falta de datos reales desde mayo 2025.\n"
        "La curva 'Real' refleja solo ingresos hasta abril.",
        fontsize=9, color="gray", ha='center', va='center', transform=plt.gcf().transFigure
    )

    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
import numpy as np

# Funciones de análisis básico
def mostrar_informacion(df):
    print("Primeras filas del DataFrame:")
    print(df.head())
    print("\nResumen estadístico:")
    print(df.describe())
    print("\nInformación del DataFrame:")
    print(df.info())

def mediana_o_media(df, columna):
    mediana = df[columna].median()
    media = df[columna].mean()
    print(f"Mediana de {columna}: {mediana:.2f}")
    print(f"Media de {columna}: {media:.2f}")

# Gráfico de promedio con etiquetas
def grafico_plot_promedio_con_valores_por_range(columna_categoria, columna_valor, tipo, titulo, ylabel, df, plt, lista_agno):
    import pandas as pd
    import matplotlib.ticker as mtick

    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df[df["Fecha"].dt.year.isin(lista_agno)]

    promedio = (
        df.groupby(columna_categoria)[columna_valor]
          .mean()
          .sort_values(ascending=False)
    )

    ax = promedio.plot(kind=tipo, title=titulo)
    ax.set_ylabel(ylabel)

    # Etiquetas sobre las barras con formato $ y miles con punto
    for contenedor in ax.containers:
        etiquetas = [f"${v:,.0f}".replace(",", ".") for v in contenedor.datavalues]
        ax.bar_label(contenedor, labels=etiquetas, padding=2)

    # Eje numérico formateado como moneda
    fmt_moneda = mtick.FuncFormatter(lambda x, _: f"${x:,.0f}".replace(",", "."))
    if tipo == 'barh':
        ax.xaxis.set_major_formatter(fmt_moneda)
    else:
        ax.yaxis.set_major_formatter(fmt_moneda)

    plt.tight_layout()
    plt.show()


# Estacionalidades
def grafico_estacionalidad_por_dia_semana_range(df, columna_fecha, columna_valor, titulo, plt, lista_agno):
    import pandas as pd
    import numpy as np
    import matplotlib.ticker as mtick

    # Copia y filtrado por años
    df = df.copy()
    df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors="coerce")
    df = df[df[columna_fecha].dt.year.isin(lista_agno)]
    df = df.dropna(subset=[columna_fecha, columna_valor])

    # 0=Lun ... 6=Dom
    df["Día_Semana"] = df[columna_fecha].dt.dayofweek
    promedio_dia = (
        df.groupby("Día_Semana", as_index=True)[columna_valor]
          .mean()
          .reindex(range(7), fill_value=np.nan)  # asegura orden Lun..Dom aunque falten días
    )

    # Gráfico
    ax = promedio_dia.plot(kind="bar", title=titulo, edgecolor="black")
    ax.set_xlabel("Día_Semana")
    ax.set_ylabel("Promedio de ventas (CLP)")
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # Etiquetas SOLO arriba de las barras con $ y miles con punto
    for cont in ax.containers:
        etiquetas = []
        for v in cont.datavalues:
            try:
                etiquetas.append(("$" + f"{float(v):,.0f}").replace(",", "."))
            except (TypeError, ValueError):
                etiquetas.append("")
        ax.bar_label(cont, labels=etiquetas, padding=2, fontsize=10, label_type="edge")

    # Eje Y con separador de miles (sin $)
    fmt_miles = mtick.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", "."))
    ax.yaxis.set_major_formatter(fmt_miles)

    # Etiquetas de los días
    ax.set_xticklabels(["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"])

    plt.tight_layout()
    plt.show()
    return ax


def grafico_estacionalidad_por_dia_semana_rango(df, columna_fecha, columna_valor, titulo, plt, lista_agno):
    df = df.copy()
    df[columna_fecha] = pd.to_datetime(df[columna_fecha])
    df = df[df[columna_fecha].dt.year.isin(lista_agno)]
    df["Día_Semana"] = df[columna_fecha].dt.dayofweek
    promedio_dia = df.groupby("Día_Semana")[columna_valor].mean()
    ax = promedio_dia.plot(kind="bar", title=titulo)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f')
    plt.xticks(ticks=range(7), labels=["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"])
    plt.tight_layout()
    plt.show()









# Generación de nuevas features
def generar_nuevas_features(df, eventos_df=None, meses_estacionales=None, detectar_meses_fn=None, anios=None):
    """
    Enriquecer DataFrame con columnas adicionales para modelado predictivo:
    - Estacionalidad temporal
    - Flags de descuento, premium, feriado
    - Clasificación BCG (basada en media o mediana según sesgo)
    - Detección automática de meses estacionales basada en outliers altos

    Parámetros:
    - df: DataFrame principal
    - eventos_df: DataFrame con columnas 'Fecha' y 'Evento' (opcional)
    - meses_estacionales: lista de meses o None
    - detectar_meses_fn: función para detectar meses si no se pasa la lista
    - anios: lista de años de referencia (si None → se usa [2023, 2024])

    Retorna:
    - DataFrame enriquecido
    """
    import pandas as pd
    from scipy.stats import skew

    if anios is None:
        anios = [2023, 2024]

    def umbral_media_o_mediana_con_analisis(df_interno, columna):
        datos = df_interno[columna].dropna()
        sesgo = skew(datos)
        if abs(sesgo) < 0.5:
            valor = datos.mean()
            interpretacion = (
                f"📊 {columna}: Distribución aproximadamente simétrica (sesgo = {sesgo:.2f}) → se usó **media** "
                f"({valor:.2f}) como umbral."
            )
        else:
            valor = datos.median()
            lado = "positiva (derecha)" if sesgo > 0 else "negativa (izquierda)"
            interpretacion = (
                f"📊 {columna}: Distribución sesgada {lado} (sesgo = {sesgo:.2f}) → se usó **mediana** "
                f"({valor:.2f}) como umbral para evitar el efecto de valores extremos."
            )
        print(interpretacion)
        return valor

    # Validación
    if "Fecha" not in df.columns:
        raise ValueError("La columna 'Fecha' es obligatoria.")

    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    # Detección automática de meses estacionales
    if meses_estacionales is None and detectar_meses_fn is not None:
        print("🔍 Detectando meses estacionales automáticamente...")
        meses_estacionales = detectar_meses_fn(
            df,
            fecha_col="Fecha",
            valor_col="Ingreso_Item",
            anios=anios,
            eventos_df=eventos_df
        )

    # Estacionalidad temporal
    df["Día del Año"] = df["Fecha"].dt.dayofyear
    df["Mes"] = df["Fecha"].dt.month
    df["Semana_Año"] = df["Fecha"].dt.isocalendar().week
    df["Fin_de_Semana"] = df["Fecha"].dt.weekday >= 5
    df["Es_Estacional"] = df["Mes"].isin(meses_estacionales if meses_estacionales else [])

    # Flags de precios
    if "Precio" in df.columns and "Precio_Original" in df.columns:
        df["Descuento_Aplicado"] = (df["Precio_Original"] - df["Precio"]) > 0
    else:
        df["Descuento_Aplicado"] = False

    if "Precio" in df.columns:
        umbral_premium = df["Precio"].quantile(0.75)
        df["Producto_Premium"] = df["Precio"] > umbral_premium
    else:
        df["Producto_Premium"] = False

    # Eventos y feriados
    if eventos_df is not None:
        eventos_df = eventos_df.copy()
        eventos_df["Fecha"] = pd.to_datetime(eventos_df["Fecha"], dayfirst=True)
        df["Dia_Feriado"] = df["Fecha"].isin(eventos_df["Fecha"])
        df = df.merge(eventos_df[["Fecha", "Evento"]], on="Fecha", how="left")
        df["Evento"] = df["Evento"].fillna("Normal")
    else:
        df["Dia_Feriado"] = False
        df["Evento"] = "Normal"

    # Clasificación BCG
    if "Ingreso_Item" in df.columns and "Cantidad" in df.columns:
        print("\n🔎 Clasificación BCG:")

        ingreso_umbral = umbral_media_o_mediana_con_analisis(df, "Ingreso_Item")
        cantidad_umbral = umbral_media_o_mediana_con_analisis(df, "Cantidad")

        condiciones = [
            (df["Cantidad"] >= cantidad_umbral) & (df["Ingreso_Item"] >= ingreso_umbral),
            (df["Cantidad"] >= cantidad_umbral) & (df["Ingreso_Item"] < ingreso_umbral),
            (df["Cantidad"] < cantidad_umbral) & (df["Ingreso_Item"] >= ingreso_umbral),
            (df["Cantidad"] < cantidad_umbral) & (df["Ingreso_Item"] < ingreso_umbral),
        ]

        etiquetas = ["Vaca Lechera 🐄", "Hormiga 🐜", "Incógnita ❓", "Perro 🐶"]
        df["BCG_Clasificacion"] = pd.Series(index=df.index, dtype=str)

        for cond, nombre in zip(condiciones, etiquetas):
            df.loc[cond, "BCG_Clasificacion"] = nombre

        resumen_bcg = df["BCG_Clasificacion"].value_counts()
        print("\n📦 Resumen de productos por clasificación BCG:")
        for categoria, count in resumen_bcg.items():
            print(f"• {categoria}: {count} productos")
    else:
        df["BCG_Clasificacion"] = "Sin Clasificar"
        print("⚠️ No se encontró 'Ingreso_Item' y/o 'Cantidad'. No se aplicó clasificación BCG.")

    return df




def detectar_meses_estacionales_por_outliers(df, fecha_col="Fecha", valor_col="Ingreso_Item", anios=None, eventos_df=None, top_n=3):
    """
    Detecta meses estacionales basados en outliers ↑ (ventas inusualmente altas).

    Parámetros:
    - df: DataFrame
    - fecha_col: nombre de la columna de fecha
    - valor_col: nombre de la columna numérica a analizar
    - anios: lista de años a evaluar (obligatoria)
    - eventos_df: eventos (opcional)
    - top_n: cantidad de meses a retornar

    Retorna:
    - Lista de meses estacionales detectados (ej. [9, 11, 12])
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from Funciones import grafico_comparacion_estacional_dias

    if anios is None:
        raise ValueError("Debes proporcionar la lista de años a evaluar (anios=[...])")

    if eventos_df is None:
        eventos_df = pd.DataFrame(columns=["Fecha", "Evento"])

    outliers_df = grafico_comparacion_estacional_dias(
        df=df,
        columna_fecha=fecha_col,
        columna_objetivo=valor_col,
        años_a_comparar=anios,
        titulo=None,
        eventos=eventos_df,
        mostrar_texto_outliers=False,
        guardar_png=False
    )

    if outliers_df.empty:
        print("⚠️ No se detectaron outliers para los años indicados.")
        return []

    outliers_df["Mes"] = pd.to_datetime(outliers_df["Fecha"]).dt.month
    meses_top = (
        outliers_df[outliers_df["Tipo"] == "↑ alto"]
        ["Mes"]
        .value_counts()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )

    print(f"📅 Meses estacionales detectados por outliers ↑: {meses_top}")
    return meses_top


# ========== NUEVO: Modelo XGBoost ==========
def entrenar_modelo_xgboost(df, columnas_features, columna_target, lista_agno, columna_fecha="Fecha"):
    import xgboost as xgb
    df = df.copy()
    df[columna_fecha] = pd.to_datetime(df[columna_fecha])
    df = df[df[columna_fecha].dt.year.isin(lista_agno)]
    df_encoded = pd.get_dummies(df[columnas_features])
    X = df_encoded
    y = df[columna_target]
    modelo = xgb.XGBRegressor(random_state=42)
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"XGBoost - MAE: {mae:.2f}, R²: {r2:.2f}")
    return modelo, mae, r2, y_pred




def grafico_estacionalidad_semanal_por_agno_rango(df, columna_fecha, columna_valor, lista_agno, titulo, plt):
    df = df.copy()
    df[columna_fecha] = pd.to_datetime(df[columna_fecha])
    df = df[df[columna_fecha].dt.year.isin(lista_agno)]
    df["Semana"] = df[columna_fecha].dt.isocalendar().week
    df["Año"] = df[columna_fecha].dt.year

    promedio = df.groupby(["Año", "Semana"])[columna_valor].mean().reset_index()

    plt.figure(figsize=(12, 6))
    for anio in lista_agno:
        subset = promedio[promedio["Año"] == anio]
        plt.plot(subset["Semana"], subset[columna_valor], label=str(anio), linewidth=1.8)

    plt.title(titulo)
    plt.xlabel("Semana del Año")
    plt.ylabel("Promedio de ventas")
    plt.grid(True)
    plt.legend(title="Año")
    plt.tight_layout()
    plt.show()


def grafico_baseball_bat_ingresos_por_anio(df, grupo_col, real_col, pred_col, titulo, anios, fecha_col, plt_mod):
    """
    Gráfico estilo 'baseball bat' (lollipop chart vertical): compara ingreso real vs. predicho por grupo.
    Pelotitas grandes tipo ⚫ (s=150) para representar cada valor.

    Parámetros:
    - df: DataFrame de entrada
    - grupo_col: columna agrupadora (ej. 'Categoría' o 'Método_Pago')
    - real_col: columna con ingresos reales (ej. 'Ingreso_Item')
    - pred_col: columna con ingresos predichos (ej. 'Prediccion_Ingreso_*')
    - titulo: título del gráfico
    - anios: lista de años a evaluar (ej. [2023, 2024])
    - fecha_col: columna con fechas
    - plt_mod: módulo matplotlib.pyplot
    """
    import pandas as pd

    def abreviar(valor):
        if valor >= 1_000_000:
            return f"${valor / 1_000_000:.2f}M"
        elif valor >= 1_000:
            return f"${valor / 1_000:.2f}K"
        else:
            return f"${valor:.2f}"

    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df_filtrado = df[df[fecha_col].dt.year.isin(anios)]

    resumen = df_filtrado.groupby(grupo_col)[[real_col, pred_col]].sum().reset_index()
    resumen = resumen.sort_values(real_col)

    x = range(len(resumen))
    plt_mod.figure(figsize=(9, 6))
    offset = 0.2

    for i in x:
        y = resumen[grupo_col].iloc[i]
        val_real = resumen[real_col].iloc[i]
        val_pred = resumen[pred_col].iloc[i]

        x_real = i - offset
        x_pred = i + offset

        # Líneas verticales y pelotitas negras (estilo bat)
        plt_mod.plot([x_real, x_real], [0, val_real], color='skyblue', linewidth=2)
        plt_mod.plot([x_pred, x_pred], [0, val_pred], color='orange', linewidth=2, linestyle='--')

        plt_mod.scatter(x_real, val_real, color='skyblue', s=150, label='Real' if i == 0 else "")
        plt_mod.scatter(x_pred, val_pred, color='orange', s=150, label='Predicho' if i == 0 else "")

        plt_mod.text(x_real, val_real + 70000, abreviar(val_real), ha='center', fontsize=9)
        plt_mod.text(x_pred, val_pred + 70000, abreviar(val_pred), ha='center', fontsize=9)

    plt_mod.xticks(x, resumen[grupo_col], rotation=45)
    plt_mod.ylabel("Ingreso")
    plt_mod.title(titulo)
    plt_mod.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt_mod.legend()
    plt_mod.tight_layout()
    plt_mod.show()

def grafico_lineas_ingresos_por_anios_filtro(
    df, grupo_col, real_col, pred_col, titulo, anios, fecha_col, plt_mod,
    frecuencia="A", categorias_filtrar=None
):
    """
    Gráfico de líneas: compara ingresos reales vs predichos, por grupo y período seleccionado.

    Parámetros:
    - df: DataFrame con datos
    - grupo_col: columna para agrupar (ej: "Categoría")
    - real_col: nombre de la columna con valores reales
    - pred_col: nombre de la columna con valores predichos
    - titulo: título del gráfico
    - anios: lista de años a incluir (ej: [2023, 2024])
    - fecha_col: nombre de la columna con fechas
    - plt_mod: módulo matplotlib.pyplot
    - frecuencia: "A" = anual, "M" = mensual, "Q" = trimestral
    - categorias_filtrar: lista de categorías específicas a incluir (opcional)
    """

    import pandas as pd

    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col], errors='coerce')
    df = df[df[fecha_col].dt.year.isin(anios)]

    # Agrupación temporal
    if frecuencia == "A":
        df["Periodo"] = df[fecha_col].dt.year.astype(str)
    elif frecuencia == "M":
        df["Periodo"] = df[fecha_col].dt.month
    elif frecuencia == "Q":
        df["Periodo"] = df[fecha_col].dt.to_period("Q").dt.quarter
    else:
        raise ValueError("Frecuencia no válida. Usa 'A', 'M' o 'Q'.")

    # Agrupar y resumir
    resumen = df.groupby(["Periodo", grupo_col])[[real_col, pred_col]].sum().reset_index()

    # Filtro por categoría (si aplica)
    if categorias_filtrar:
        resumen = resumen[resumen[grupo_col].isin(categorias_filtrar)]

    # Etiquetas de meses
    if frecuencia == "M":
        meses = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
            5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
            9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }
        resumen["Periodo"] = resumen["Periodo"].map(meses)

    plt_mod.figure(figsize=(12, 6))

    categorias = resumen[grupo_col].unique()
    for cat in categorias:
        datos = resumen[resumen[grupo_col] == cat]
        plt_mod.plot(datos["Periodo"], datos[real_col], marker='o', label=f"{cat} - Real", linestyle='-')
        plt_mod.plot(datos["Periodo"], datos[pred_col], marker='o', label=f"{cat} - Predicho", linestyle='--')

    plt_mod.title(titulo)
    plt_mod.xlabel("Periodo")
    plt_mod.ylabel("Ingreso")
    plt_mod.xticks(rotation=45)
    plt_mod.legend()
    plt_mod.grid(True, linestyle='--', alpha=0.5)
    plt_mod.tight_layout()
    plt_mod.show()


def calcular_q_optimo_con_stock_seguridad_v4(
    df_productos,
    df_envios,
    columna_producto="Nombre_Producto_Real",
    columna_fecha="Fecha",
    columna_fecha_envio="Fecha_Envío",
    columna_fecha_entrega="Fecha_Entrega_Final",
    columna_cantidad="Cantidad",
    columna_sucursal=None,
    costos_orden={},
    costos_mantencion={},
    incluir_stock_seguridad=True,
    periodo="M",  # "M" (mes), "Q" (trimestre), "Y" (año)
    filtro_periodo=None,
    imprimir_heatmap=True
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import shapiro

    df_envios = df_envios.copy()
    df_productos = df_productos.copy()

    # Si no existe columna de producto en envíos, intentar unir por ID
    if columna_producto not in df_envios.columns:
        if "ID_Producto" in df_envios.columns and "ID_Producto" in df_productos.columns:
            df_envios = df_envios.merge(
                df_productos[["ID_Producto", columna_producto]].drop_duplicates(),
                on="ID_Producto", how="left"
            )

    # Asegurar tipo datetime
    df_envios[columna_fecha_envio] = pd.to_datetime(df_envios[columna_fecha_envio], errors='coerce')
    df_envios[columna_fecha_entrega] = pd.to_datetime(df_envios[columna_fecha_entrega], errors='coerce')
    df_productos[columna_fecha] = pd.to_datetime(df_productos[columna_fecha], errors='coerce')

    # Calcular lead time
    df_envios["Lead_Time_Dias"] = (df_envios[columna_fecha_entrega] - df_envios[columna_fecha_envio]).dt.days
    df_envios = df_envios.dropna(subset=["Lead_Time_Dias"])

    # Crear columna de periodo
    df_productos["Fecha_Periodo"] = df_productos[columna_fecha].dt.to_period(periodo).dt.start_time

    # Aplicar filtro si se especifica
    if filtro_periodo is not None:
        fechas_filtrar = pd.to_datetime(filtro_periodo)
        df_productos = df_productos[df_productos["Fecha_Periodo"].isin(fechas_filtrar)]

    productos = df_productos[columna_producto].unique()
    resultados = []

    for producto in productos:
        df_prod = df_productos[df_productos[columna_producto] == producto]
        df_env = df_envios[df_envios[columna_producto] == producto]

        if df_prod.empty or df_env.empty:
            continue

        for fecha_periodo, df_p in df_prod.groupby("Fecha_Periodo"):
            demanda_diaria = df_p.groupby(df_p[columna_fecha].dt.date)[columna_cantidad].sum()

            # Método estadístico para D
            stat_d, p_d = shapiro(demanda_diaria)
            usar_mediana_d = p_d < 0.05
            d = demanda_diaria.median() if usar_mediana_d else demanda_diaria.mean()
            texto_d = "Mediana" if usar_mediana_d else "Media"

            # Lead Time
            lead_times = df_env["Lead_Time_Dias"]
            stat_lt, p_lt = shapiro(lead_times)
            usar_mediana_lt = p_lt < 0.05
            lead_time = lead_times.median() if usar_mediana_lt else lead_times.mean()
            texto_lt = "Mediana" if usar_mediana_lt else "Media"

            # Costos
            s = costos_orden.get(producto, 100)
            h = costos_mantencion.get(producto, 10)

            # Stock de seguridad
            stock_seguridad = 0
            if incluir_stock_seguridad:
                desviacion_d = demanda_diaria.std()
                desviacion_lt = lead_times.std()
                stock_seguridad = 1.65 * np.sqrt((desviacion_d ** 2 * lead_time) + (d ** 2 * desviacion_lt ** 2))

            Q = np.sqrt((2 * d * s) / h)

            resultados.append({
                "Producto": producto,
                "Periodo": fecha_periodo.strftime("%Y-%m"),
                "Lead Time": round(lead_time, 2),
                "Método LT": texto_lt,
                "Demanda Diaria": round(d, 2),
                "Método D": texto_d,
                "Costo Orden (S)": s,
                "Costo Mantención (H)": h,
                "Stock Seguridad": round(stock_seguridad, 2),
                "Q Óptimo": round(Q, 2)
            })

    df_resultados = pd.DataFrame(resultados)

    if imprimir_heatmap and not df_resultados.empty:
        pivot = df_resultados.pivot(index="Producto", columns="Periodo", values="Q Óptimo")
        pivot = pivot[sorted(pivot.columns)]  # Ordenar cronológicamente

        plt.figure(figsize=(14, 7))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "Q Óptimo"})
        plt.title("Q Óptimo por Producto y Periodo", fontsize=14)
        plt.xlabel("Periodo")
        plt.ylabel("Producto")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    return df_resultados


def analizar_lead_time_avanzado(
    df_envios,
    df_productos,
    columna_producto="Nombre_Producto_Real",
    columna_id_producto="ID_Producto",
    columna_fecha_envio="Fecha_Envío",
    columna_fecha_entrega="Fecha_Entrega_Final",
    columna_fecha="Fecha",
    filtro_periodo=None,
    periodo="M",  # "M" para mes, "Q" para trimestre, "Y" para año
    mostrar_boxplot=False,
    mostrar_heatmap=True
):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import shapiro

    df_envios = df_envios.copy()
    df_productos = df_productos.copy()

    # --- Asegurar columnas necesarias
    if columna_producto not in df_envios.columns:
        if columna_id_producto in df_envios.columns and columna_id_producto in df_productos.columns:
            df_envios = df_envios.merge(
                df_productos[[columna_id_producto, columna_producto]].drop_duplicates(),
                on=columna_id_producto,
                how="left"
            )

    # --- Procesar fechas
    df_envios[columna_fecha_envio] = pd.to_datetime(df_envios[columna_fecha_envio], errors='coerce')
    df_envios[columna_fecha_entrega] = pd.to_datetime(df_envios[columna_fecha_entrega], errors='coerce')

    # --- Calcular lead time
    df_envios["Lead_Time_Dias"] = (df_envios[columna_fecha_entrega] - df_envios[columna_fecha_envio]).dt.days
    df_envios.dropna(subset=["Lead_Time_Dias"], inplace=True)

    # --- Crear columna de periodo para filtrar
    df_envios["Periodo"] = df_envios[columna_fecha_envio].dt.to_period(periodo).dt.start_time

    # --- Filtro de periodo si aplica
    if filtro_periodo:
        fechas_filtrar = pd.to_datetime(filtro_periodo)
        df_envios = df_envios[df_envios["Periodo"].isin(fechas_filtrar)]

    # --- Agrupación
    resultados = []
    productos = df_envios[columna_producto].dropna().unique()

    for producto in productos:
        df_p = df_envios[df_envios[columna_producto] == producto]

        if df_p.empty:
            continue

        for periodo_actual, df_gp in df_p.groupby("Periodo"):
            lead_times = df_gp["Lead_Time_Dias"]

            stat, p_valor = shapiro(lead_times)
            usar_mediana = p_valor < 0.05

            lead_resultado = lead_times.median() if usar_mediana else lead_times.mean()
            metodo_usado = "Mediana" if usar_mediana else "Media"

            resultados.append({
                "Producto": producto,
                "Periodo": periodo_actual.strftime("%Y-%m") if periodo != "Y" else periodo_actual.strftime("%Y"),
                "Lead Time": round(lead_resultado, 2),
                "Método": metodo_usado
            })

    df_resultados = pd.DataFrame(resultados)

    # --- HEATMAP
    if mostrar_heatmap and not df_resultados.empty:
        pivot = df_resultados.pivot(index="Producto", columns="Periodo", values="Lead Time")
        pivot = pivot.reindex(sorted(pivot.index), axis=0)  # ordenar productos
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)  # ordenar periodos

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrBr", cbar_kws={"label": "Lead Time (días)"})
        plt.title("Lead Time por Producto y Periodo", fontsize=14)
        plt.xlabel("Periodo")
        plt.ylabel("Producto")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # --- BOXPLOT
    if mostrar_boxplot:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_envios, x=columna_producto, y="Lead_Time_Dias", palette="Set3")
        plt.title("Distribución de Lead Time por Producto", fontsize=14)
        plt.xlabel("Producto")
        plt.ylabel("Lead Time (días)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    return df_resultados


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
import plotly.graph_objects as go


def calcular_q_optimo_para_dashboard(
    df_productos,
    df_envios,
    columna_producto="Nombre_Producto_Real",
    columna_fecha="Fecha",
    columna_fecha_envio="Fecha_Envío",
    columna_fecha_entrega="Fecha_Entrega_Final",
    columna_cantidad="Cantidad",
    columna_sucursal=None,
    costos_orden={},
    costos_mantencion={},
    incluir_stock_seguridad=True,
    imprimir_heatmap=False,
    imprimir_dashboard=True,
    mostrar_tabla=True
):
    df_envios = df_envios.copy()
    df_productos = df_productos.copy()

    # Asegurar que existan las columnas necesarias en df_envios
    if columna_producto not in df_envios.columns:
        if "ID_Producto" in df_envios.columns and "ID_Producto" in df_productos.columns:
            df_envios = df_envios.merge(
                df_productos[["ID_Producto", columna_producto]].drop_duplicates(),
                on="ID_Producto",
                how="left"
            )

    # Asegurar formato de fechas
    df_envios[columna_fecha_envio] = pd.to_datetime(df_envios[columna_fecha_envio], errors='coerce')
    df_envios[columna_fecha_entrega] = pd.to_datetime(df_envios[columna_fecha_entrega], errors='coerce')
    df_productos[columna_fecha] = pd.to_datetime(df_productos[columna_fecha], errors='coerce')

    # Calcular Lead Time
    df_envios["Lead_Time_Dias"] = (df_envios[columna_fecha_entrega] - df_envios[columna_fecha_envio]).dt.days
    df_envios = df_envios.dropna(subset=["Lead_Time_Dias"])

    productos = df_productos[columna_producto].unique()
    resultados = []

    for producto in productos:
        df_prod = df_productos[df_productos[columna_producto] == producto]
        df_env = df_envios[df_envios[columna_producto] == producto]

        if df_prod.empty or df_env.empty:
            continue

        df_prod = df_prod.copy()
        df_prod["Fecha"] = pd.to_datetime(df_prod[columna_fecha])
        df_prod["Periodo"] = df_prod["Fecha"].dt.to_period("M").dt.to_timestamp()

        demanda_por_fecha = df_prod.groupby("Fecha")[columna_cantidad].sum()
        stat_d, p_d = shapiro(demanda_por_fecha)
        usar_mediana_d = p_d < 0.05
        d = demanda_por_fecha.median() if usar_mediana_d else demanda_por_fecha.mean()
        texto_d = "Mediana" if usar_mediana_d else "Media"

        lead_times = df_env["Lead_Time_Dias"]
        stat_lt, p_lt = shapiro(lead_times)
        usar_mediana_lt = p_lt < 0.05
        lead_time = lead_times.median() if usar_mediana_lt else lead_times.mean()
        texto_lt = "Mediana" if usar_mediana_lt else "Media"

        s = costos_orden.get(producto, 100)
        h = costos_mantencion.get(producto, 10)

        stock_seguridad = 0
        if incluir_stock_seguridad:
            desviacion_d = demanda_por_fecha.std()
            desviacion_lt = lead_times.std()
            stock_seguridad = 1.65 * np.sqrt((desviacion_d ** 2 * lead_time) + (d ** 2 * desviacion_lt ** 2))

        for periodo, grupo in df_prod.groupby("Periodo"):
            demanda_por_dia = grupo.groupby(grupo["Fecha"].dt.date)[columna_cantidad].sum()
            d_local = demanda_por_dia.median() if usar_mediana_d else demanda_por_dia.mean()
            Q = np.sqrt((2 * d_local * s) / h)
            resultados.append({
                "Producto": producto,
                "Fecha": periodo,
                "Q_Óptimo": round(Q, 2),
                "Lead Time": round(lead_time, 2),
                "Método LT": texto_lt,
                "Demanda Diaria": round(d_local, 2),
                "Método D": texto_d,
                "Costo Orden (S)": s,
                "Costo Mantención (H)": h,
                "Stock Seguridad": round(stock_seguridad, 2),
            })

    df_resultado = pd.DataFrame(resultados)

    if mostrar_tabla:
        print(df_resultado)

    if imprimir_heatmap:
        pivot = df_resultado.pivot(index="Producto", columns="Fecha", values="Q_Óptimo")
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Q Óptimo"})
        plt.title("Q Óptimo por Producto y Periodo")
        plt.xlabel("Periodo")
        plt.ylabel("Producto")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    if imprimir_dashboard:
        df_resultado["Año"] = df_resultado["Fecha"].dt.year
        df_resultado["Mes"] = df_resultado["Fecha"].dt.month
        df_resultado["Trimestre"] = df_resultado["Fecha"].dt.to_period("Q")
        df_resultado = df_resultado.sort_values(by="Fecha")

        heatmap_data = df_resultado.pivot_table(
            values="Q_Óptimo",
            index="Producto",
            columns="Fecha",
            aggfunc="mean"
        )

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns.strftime("%b %Y"),
            y=heatmap_data.index,
            colorscale='YlGnBu',
            colorbar=dict(title='Q Óptimo'),
            hovertemplate='Producto: %{y}<br>Periodo: %{x}<br>Q Óptimo: %{z:.2f}<extra></extra>',
            text=heatmap_data.values,
            texttemplate="%{text:.2f}",
        ))

        fig.update_layout(
            title="Heatmap Dinámico: Q Óptimo por Producto y Periodo",
            xaxis_title="Periodo",
            yaxis_title="Producto",
            xaxis_tickangle=-45,
            autosize=True,
            margin=dict(l=80, r=20, t=60, b=80)
        )

        fig.show()

    return df_resultado


def calcular_q_optimo_con_dashboard_vx(
    df_productos,
    df_envios,
    columna_producto="Nombre_Producto_Real",
    columna_fecha="Fecha",
    columna_fecha_envio="Fecha_Envío",
    columna_fecha_entrega="Fecha_Entrega_Final",
    columna_cantidad="Cantidad",
    columna_sucursal=None,
    costos_orden={},
    costos_mantencion={},
    incluir_stock_seguridad=True,
    imprimir_dashboard=True
):
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from scipy.stats import shapiro

    df_productos = df_productos.copy()
    df_envios = df_envios.copy()

    # Formateo de fechas
    df_productos[columna_fecha] = pd.to_datetime(df_productos[columna_fecha], errors="coerce")
    df_envios[columna_fecha_envio] = pd.to_datetime(df_envios[columna_fecha_envio], errors="coerce")
    df_envios[columna_fecha_entrega] = pd.to_datetime(df_envios[columna_fecha_entrega], errors="coerce")

    # Si no tiene nombre producto, hacer merge
    if columna_producto not in df_envios.columns and "ID_Producto" in df_envios.columns:
        df_envios = df_envios.merge(
            df_productos[["ID_Producto", columna_producto]].drop_duplicates(),
            on="ID_Producto", how="left"
        )

    # Lead Time
    df_envios["Lead_Time_Dias"] = (df_envios[columna_fecha_entrega] - df_envios[columna_fecha_envio]).dt.days
    df_envios = df_envios.dropna(subset=["Lead_Time_Dias"])

    # Segmentos de tiempo
    df_productos["Año"] = df_productos[columna_fecha].dt.year
    df_productos["Mes"] = df_productos[columna_fecha].dt.strftime("%b %Y")  # Ej: Ene 2023
    df_productos["Trimestre"] = df_productos[columna_fecha].dt.to_period("Q").astype(str)

    resultados = []
    productos = df_productos[columna_producto].unique()

    for producto in productos:
        df_prod = df_productos[df_productos[columna_producto] == producto]
        df_env = df_envios[df_envios[columna_producto] == producto]

        if df_prod.empty or df_env.empty:
            continue

        for (año, trimestre, mes), grupo in df_prod.groupby(["Año", "Trimestre", "Mes"]):
            demanda_diaria = grupo.groupby(grupo[columna_fecha].dt.date)[columna_cantidad].sum()

            stat_d, p_d = shapiro(demanda_diaria)
            usar_mediana_d = p_d < 0.05
            d = demanda_diaria.median() if usar_mediana_d else demanda_diaria.mean()
            texto_d = "Mediana" if usar_mediana_d else "Media"

            lead_times = df_env["Lead_Time_Dias"]
            stat_lt, p_lt = shapiro(lead_times)
            usar_mediana_lt = p_lt < 0.05
            lead_time = lead_times.median() if usar_mediana_lt else lead_times.mean()
            texto_lt = "Mediana" if usar_mediana_lt else "Media"

            s = costos_orden.get(producto, 100)
            h = costos_mantencion.get(producto, 10)

            stock_seguridad = 0
            if incluir_stock_seguridad:
                desviacion_d = demanda_diaria.std()
                desviacion_lt = lead_times.std()
                stock_seguridad = 1.65 * np.sqrt((desviacion_d ** 2 * lead_time) + (d ** 2 * desviacion_lt ** 2))

            Q = np.sqrt((2 * d * s) / h)

            resultados.append({
                "Producto": producto,
                "Año": año,
                "Trimestre": trimestre,
                "Mes": mes,
                "Lead Time": round(lead_time, 2),
                "Método LT": texto_lt,
                "Demanda Diaria": round(d, 2),
                "Método D": texto_d,
                "Costo Orden (S)": s,
                "Costo Mantención (H)": h,
                "Stock Seguridad": round(stock_seguridad, 2),
                "Q Óptimo": round(Q, 2)
            })

    df_resultados = pd.DataFrame(resultados)

    # Mostrar dashboard con botones dinámicos
    if imprimir_dashboard and not df_resultados.empty:
        fig = px.sunburst(
            df_resultados,
            path=["Año", "Trimestre", "Mes", "Producto"],
            values="Q Óptimo",
            color="Q Óptimo",
            color_continuous_scale="Teal",
            title="Visualización jerárquica de Q Óptimo"
        )
        fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
        fig.show()

    return df_resultados




def analizar_lead_time_segmentado(df_envios, df_productos):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import shapiro

    # Copias limpias
    df_envios = df_envios.copy()
    df_productos = df_productos.copy()

    # Unir nombre de producto
    if "Nombre_Producto_Real" not in df_envios.columns:
        df_envios = df_envios.merge(
            df_productos[["ID_Producto", "Nombre_Producto_Real"]].drop_duplicates(),
            on="ID_Producto",
            how="left"
        )

    # Fechas
    df_envios["Fecha_Envío"] = pd.to_datetime(df_envios["Fecha_Envío"], errors="coerce")
    df_envios["Fecha_Entrega_Final"] = pd.to_datetime(df_envios["Fecha_Entrega_Final"], errors="coerce")
    df_envios["Fecha"] = df_envios["Fecha_Envío"]  # para filtros generales

    # Lead Time
    df_envios["Lead_Time_Dias"] = (df_envios["Fecha_Entrega_Final"] - df_envios["Fecha_Envío"]).dt.days
    df_envios = df_envios.dropna(subset=["Lead_Time_Dias"])

    # Segmentadores
    df_envios["Año"] = df_envios["Fecha"].dt.year
    df_envios["Mes"] = df_envios["Fecha"].dt.month
    df_envios["Trimestre"] = df_envios["Fecha"].dt.quarter

    regiones = df_envios["Centro_Distribucion"].dropna().unique().tolist()
    ciudades = df_envios["Ciudad_Destino"].dropna().unique().tolist()
    años = sorted(df_envios["Año"].unique().tolist())

    import ipywidgets as widgets
    from IPython.display import display

    sel_region = widgets.SelectMultiple(options=regiones, value=regiones[:1], description="Centro:")
    sel_ciudad = widgets.SelectMultiple(options=ciudades, value=ciudades[:1], description="Ciudad:")
    sel_anios = widgets.SelectMultiple(options=años, value=[años[-1]], description="Años:")
    sel_mes = widgets.IntRangeSlider(value=[1,12], min=1, max=12, step=1, description="Mes:")
    sel_trim = widgets.IntRangeSlider(value=[1,4], min=1, max=4, step=1, description="Trimestre:")

    def actualizar_graficos(region, ciudad, anios, mes_range, trim_range):
        df_filtro = df_envios[
            (df_envios["Centro_Distribucion"].isin(region)) &
            (df_envios["Ciudad_Destino"].isin(ciudad)) &
            (df_envios["Año"].isin(anios)) &
            (df_envios["Mes"].between(mes_range[0], mes_range[1])) &
            (df_envios["Trimestre"].between(trim_range[0], trim_range[1]))
        ]

        if df_filtro.empty:
            print("⚠️ No hay datos para los filtros seleccionados.")
            return

        # 📊 Gráfico Violin
        plt.figure(figsize=(10, 5))
        sns.violinplot(data=df_filtro, x="Nombre_Producto_Real", y="Lead_Time_Dias")
        plt.title("Distribución de Lead Time por Producto")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # 🔥 Heatmap
        pivot = df_filtro.groupby(["Nombre_Producto_Real", "Mes"])["Lead_Time_Dias"].mean().unstack()
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "Lead Time"})
        plt.title("Heatmap de Lead Time promedio por Producto y Mes")
        plt.xlabel("Mes")
        plt.ylabel("Producto")
        plt.tight_layout()
        plt.show()

    ui = widgets.VBox([sel_region, sel_ciudad, sel_anios, sel_mes, sel_trim])
    out = widgets.interactive_output(
        actualizar_graficos,
        {
            "region": sel_region,
            "ciudad": sel_ciudad,
            "anios": sel_anios,
            "mes_range": sel_mes,
            "trim_range": sel_trim
        }
    )
    display(ui, out)


# ========= MÉTRICAS (WAPE, sMAPE, MASE, Bias) =========
import numpy as _np
import pandas as _pd
from sklearn.model_selection import train_test_split as _tts

def _calcular_metricas_adicionales(y_true, y_pred, eps=1e-8, fallback_cero=True):
    y_true = _np.asarray(y_true, dtype=float)
    y_pred = _np.asarray(y_pred, dtype=float)

    mae   = float(_np.mean(_np.abs(y_true - y_pred)))  # para MASE
    wape  = float(_np.sum(_np.abs(y_true - y_pred)) / (_np.sum(_np.abs(y_true)) + eps))
    smape = float(_np.mean(2.0 * _np.abs(y_true - y_pred) / (_np.abs(y_true) + _np.abs(y_pred) + eps)))
    bias  = float(_np.mean(y_pred - y_true))
    bias_pct = float(_np.sum(y_pred - y_true) / (_np.sum(_np.abs(y_true)) + eps))

    # MASE (naïve m=1) sobre el set de TEST; con fallback si el denominador ≈ 0
    if y_true.size >= 2:
        denom = float(_np.mean(_np.abs(y_true[1:] - y_true[:-1])))
        if _np.isfinite(denom) and denom > eps:
            mase = float(mae / denom)
        else:
            mase = 0.0 if fallback_cero else float("nan")
    else:
        mase = 0.0 if fallback_cero else float("nan")

    return {"WAPE": wape, "sMAPE": smape, "MASE": mase, "Bias": bias, "Bias_pct": bias_pct}


def metricas_modelo(
    modelo, df, columnas_entrada=None,
    columna_objetivo="Ingreso_Item", años=(2023, 2024), fecha_col="Fecha",
    test_size=0.2, random_state=42, fallback_cero=True
):
    """
    Calcula WAPE, sMAPE, MASE y Bias para un modelo ya entrenado.
    - Si columnas_entrada es None, intenta leerlas de modelo.feature_names_in_.
    - Usa el mismo filtro de años y un split reproducible (random_state=42).
    """
    if columnas_entrada is None and hasattr(modelo, "feature_names_in_"):
        columnas_entrada = list(modelo.feature_names_in_)
    if columnas_entrada is None:
        raise ValueError("Debes pasar columnas_entrada o entrenar el modelo con feature_names_in_.")

    df2 = df.copy()
    df2[fecha_col] = _pd.to_datetime(df2[fecha_col], errors="coerce")
    df2 = df2[df2[fecha_col].dt.year.isin(list(años))].dropna(subset=columnas_entrada + [columna_objetivo])

    X = df2[columnas_entrada]
    y = df2[columna_objetivo]

    # mismo split para una comparación justa
    _, X_test, _, y_test = _tts(X, y, test_size=test_size, random_state=random_state)
    y_pred = modelo.predict(X_test)

    return _calcular_metricas_adicionales(y_test, y_pred, fallback_cero=fallback_cero)
# ========= FIN MÉTRICAS =========








