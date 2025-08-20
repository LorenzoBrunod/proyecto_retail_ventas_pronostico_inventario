# proyecto_retail_ventas_pronostico_inventario
EDA y modelado de ventas (base vs enriquecido) con métricas WAPE/sMAPE/MASE y visualizaciones en CLP. Traducción a inventario: EOQ (Q*), Stock de Seguridad, ROP y gobierno de Lead Time. Incluye cierre metodológico, KPIs y cadencia S&OP.

# Ejemplo de Análisis Exploratorio de Datos 

## Análisis de Ventas, Predicción e Inventario (Retail | CLP)

## Introducción
El comportamiento de la demanda en retail está guiado por **estacionalidad**, eventos comerciales y el **mix de categorías**. Este informe integra **todos los gráficos** y **resultados** generados por `codigo_principal.py` y `C_Principal.py`, y baja a decisiones operativas: **Q óptimo (EOQ)**, **Stock de Seguridad (SS)**, **Punto de Reorden (ROP)** y **Lead Time**.  
> **Moneda:** CLP. En gráficos, `$` solo en etiquetas sobre barras; el eje indica la unidad.

---

## Objetivos del Proyecto
1_ Analizar patrones y correlaciones en ventas (día, semana, mes).  
2_ Comparar desempeño por **categoría** y **método de pago**.  
3_ Evaluar ajuste de los **modelos base y enriquecidos** (2023–2025).  
4_ Traducir el análisis a decisiones de **inventario** (SS/ROP/EOQ) y **Lead Time**.

---

## Requisitos
```bash
python >= 3.9
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl
```

---

## Estructura del repositorio (código en carpetas)
```
.
├─ README.md
├─ Base_Empresa_BigData_Limpio.xlsx   # dataset limpio (déjalo en la raíz)
├─ src/
│  ├─ C_Principal.py
│  ├─ codigo_principal.py
│  └─ Funciones.py
└─ docs/
   └─ img/
      └─ inventario/                  # opcional: si guardas PNG de EOQ/SS/ROP/LT
```
> Respeta mayúsculas/minúsculas tal como arriba. Los scripts esperan el Excel en la **raíz**.

---

## Cómo ejecutar (sin tocar el código)
### Opción A — Flujo completo
```bash
python src/codigo_principal.py
python src/C_Principal.py
```
### Opción B — Solo análisis/gráficos/modelos
```bash
python src/C_Principal.py
```
Los scripts:
- Cargan **Base_Empresa_BigData_Limpio.xlsx**.
- Construyen agregados por **día, semana y mes**.
- Ejecutan las **funciones de `Funciones.py`** para visualizaciones, correlaciones, modelos (Random Forest) y cálculos de inventario (Q*, SS, ROP y Lead Time).  
- Imprimen métricas por consola y muestran/guardan gráficos según tu configuración.

> Visualización: usa **VS Code** o Jupyter; Matplotlib abrirá las figuras en pantalla. Si el script guarda archivos, aparecerán en `docs/img/` (y `docs/img/inventario/`).

---

## Limpieza y Transformaciones de Datos
- Tipificación de fechas a `datetime` y derivaciones: `Día del Año`, `Semana del Año`, `Día_Semana`, `Fin_de_Semana`.  
- Normalización de categorías y **método de pago**.  
- Flags de negocio: `Producto_Premium`, `Descuento_Aplicado`, `Es_Estacional`.  
- Variables para inventario: **Lead_Time_Dias** (μ, σ, percentiles) por CD×Ciudad (si disponible).

---

## Análisis de Distribución
- En agregados **diarios**: ruido con colas por eventos.  
- En **semanal/mensual**: perfil estable y repetible.  
**Conclusión:** la estacionalidad domina; para planeamiento táctico conviene trabajar semanal/mensual y elevar buffers en ventanas de evento.

---

## Justificación Técnica de Métodos Utilizados

| Método / Técnica                        | Justificación                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------|
| `.info()`, `.describe()`, `groupby()`   | Comprensión, agregaciones y cortes por tiempo/categoría.                      |
| Heatmaps de correlación                 | Relación entre Precio, Cantidad e Ingreso; evitar multicolinealidades.        |
| Líneas y barras                         | Lectura de estacionalidad y comparaciones por grupo.                          |
| RandomForest (base vs. enriquecido)     | Modelo no lineal, interpretabilidad vía importancias.                         |
| Validación temporal (*TimeSeriesSplit*) | Evitar sobreestimación por *shuffle* en series temporales.                    |
| Métricas (MAE, R², WAPE, sMAPE, MASE)   | Evaluación robusta y comparable entre categorías y periodos.                  |
| Inventario (SS, ROP, EOQ)               | Traducir pronósticos a decisiones de compra y reposición.                     |

---

## Visualizaciones y su Interpretación
> Sube **tus imágenes** a `docs/img/` con **estos nombres exactos** (los que me enviaste). Las rutas ya están listas.

### Estacionalidad
1. **Semanal (líneas, 2023–2024)**  
   ![Ventas por semana](<docs/img/Análisis Estacional de Ventas por Semana Año 2023-2024(líneas).png>)  
   **Insight:** curvas casi paralelas; estacionalidad estable con picos puntuales.  
2. **Semanal (barras, 2023–2024)**  
   ![Ventas semanales barras](<docs/img/Ánalisis Estacional de Ventas Semanales Año 2023-2024.png>)  
   **Insight:** jueves y fin de semana levemente superiores; patrón intra-semana predecible.  
3. **Diaria (líneas, 2023–2024)**  
   ![Ventas diarias](<docs/img/Análisis Estacional Diario de Ventas Año 2023-2024(líneas).png>)  
   **Insight:** ruido diario; conviene agregar a semana/mes para decisiones.  
4. **Diaria con outliers + eventos**  
   ![Outliers y eventos](<docs/img/Comparación Estacional de Ventas con outliers Diarias entre 2023-2024.png>)  
   **Insight:** Black Friday y Navidad concentran picos → **elevar SS y ROP** en ventanas de evento.  
5. **Mensual (barras)**  
   ![Estacionalidad mensual](<docs/img/Estacionalidad por mes Año 2023-2024 (Barras).png>)  
   **Insight:** máximos claros en **septiembre** y **diciembre**; planificar **stock build** previo.  
6. **Fechas con mayor estacionalidad**  
   ![Fechas clave](<docs/img/Fechas con mayor estacionalidad.png>)  
   **Insight:** días específicos explican gran parte del ingreso; gatillos para buffers logísticos.

### Mix y pagos
7. **Promedio por categoría (2023–2024)**  
   ![Promedio por categoría](<docs/img/Promedio por Categoría Año 2023-2024(barras).png>)  
   **Insight:** **Hogar/Oficina** lideran; **Electrónica/Juguetería** rezagadas.  
8. **Promedios por método de pago (2023–2024)**  
   ![Promedio por pago](<docs/img/Promedio por metodo de pago año 2023-2024.png>)  
   **Insight:** diferencias marginales; **App** apenas mayor.  
9. **Total de ingresos por método de pago (2023–2024)**  
   ![Total por pago](<docs/img/Total de Ingresos por Método de Pago Años 2023-2024 con variables elegidas.png>)  
   **Insight:** cartera de pagos diversificada; riesgo operativo bajo por canal.  
10. **Distribución $ por categoría×método de pago**  
    ![Distribución cat×pago](<docs/img/Distribución $ de Categoría por Método de Pago Año 2023-2024.png>)  
    **Insight:** los métodos no reconfiguran sustancialmente el mix por categoría.  
11. **Distribución de frecuencia de compra**  
    ![Frecuencia compra](<docs/img/Distribución de Frecuencia de Compra Año 2023-2024.png>)  
    **Insight:** base de clientes con repetición suficiente para planificar por mes.  
12. **Participación por categoría**  
    ![Participación](docs/img/Participaci%C3%B3n%20%25%20por%20Categor%C3%ADa%20de%20Producto%20A%C3%B1o%202023-2024.png) 
    **Insight:** **Oficina/Hogar** concentran el share → asignar capacidad proporcional.

### Correlaciones
13. **Precio–Cantidad–Ingreso (2023–2024) — combinación 1**  
    ![Mapa correlación 1](<docs/img/Mapa de correlación Año 2023-2024(primera combinación).png>)  
14. **Precio–Cantidad–Ingreso (2023–2024) — combinación 2**  
    ![Mapa correlación 2](<docs/img/Segunda Combinación mapa de correlación.png>)  
    **Insight:** Ingreso depende fuertemente de **Precio** y moderadamente de **Cantidad**.

### Real vs. Predicho (modelos)
15. **Por categoría — Base (2023–2024)**  
    ![Base 23-24](<docs/img/Ingreso Real vs Predicho(base) enriqueido 2023-2024.png>)  
    **Insight:** captura estacionalidad; errores en cambios bruscos.  
16. **Por categoría — Enriquecido (2023–2024)**  
    ![Enriquecido 23-24](<docs/img/Ingreso Real vs Predicho(Enriquecido) por categoría 2023_2024.png>)  
    **Insight:** menor gap gracias a señales de negocio.  
17. **Por método de pago — Lollipop Base (2023–2024)**  
    ![Lollipop Base](<docs/img/Lolliop Ingreso Real vs Predicho(Base) por Método de Pago 2023-2024.png>)  
18. **Por método de pago — Lollipop Enriquecido (2023–2024)**  
    ![Lollipop Enriquecido](<docs/img/Lollipop Ingreso Real vs Predicho(Enirquecido) por Método de peago.png>)  
    **Insight:** el enriquecido reduce diferencias residuales.  
19. **Importancia de variables — Base**  
    ![FI Base](<docs/img/Importancia de Variables modelo forest base.png>)  
20. **Importancia de variables — Enriquecido**  
    ![FI Enriq.](<docs/img/Importancia de Variables modelo eriquecido.png>)  
    **Insight:** `Día del Año` domina; el enriquecido suma señales operativas.  
21. **Por categoría — Base (2025)**  
    ![Base 2025](<docs/img/Ingreso Real vs Predicho(Base) por categoría 2025.png>)  
22. **Por categoría — (Mensual) Enriquecido 2025**  
    ![Mensual 2025 enr.](<docs/img/Ingerso Real vs Predicho enriquecido por Categoría (Mensual) 2025.png>)
23. **Por categoría — (Mensual) Real vs Predicha 2025**  
    ![Mensual 2025 real/pred](<docs/img/Cantidad Real vs Predicha por Categoría (Mneusal) 2025.png>)  

---

## Análisis / Modelado Aplicado
- **Comparativa Base vs Enriquecido:** el enriquecido reduce el **gap visual** y suaviza el **Bias** en picos.  
- **Sesgo por categoría (2025):** sub-predicción leve en **Hogar/Oficina** → criterio para **buffers de SS** diferenciados.  
- **Métricas multi-escala:** junto a MAE/R², el uso de **WAPE, sMAPE, MASE y Bias** permite comparar categorías de distinta escala.

---

## Métricas del modelo (qué reporto y cómo leerlas)
- **WAPE** = Σ|y − ŷ| / Σ|y| → error relativo agregado (**↓ mejor**).  
- **sMAPE** = mean( |y−ŷ| / ((|y|+|ŷ|)/2) ) → robusta a escala (**↓ mejor**).  
- **MASE** → error respecto a un *naive* estacional (**↓ mejor**, <1 supera al naive).  
- **MAE** → error absoluto medio (**↓ mejor**).  
- **R²** → varianza explicada (**↑ mejor**).  
- **Bias** = mean(ŷ − y) / mean(y) → signo del sesgo (0 ideal; + sobre-predice, − sub-predice).

**Criterios propuestos de aceptación:** WAPE ≤ **12%** agregado, Bias en **[−2%, +2%]**, y por categoría A ≤ **15%**.

---

## ¿Cómo ver Q*, Stock de Seguridad (SS), ROP y Lead Times?
- **Se visualizan en VS Code/Jupyter** ejecutando `python src/codigo_principal.py` o `python src/C_Principal.py`.  
- Si tus funciones guardan archivos, usa estas carpetas sugeridas (puedes crearlas):  
  - PNG **inventario** → `docs/img/inventario/`  
  - CSV **inventario** → `outputs/inventario/`

> Si aún no guardas automáticamente, igual verás los gráficos en pantalla (VS Code). Para mostrarlos en el README, súbelos luego a `docs/img/inventario/` y enlázalos.

### Lectura rápida (inventario)
- **Q\*** sube en meses pico (sep/dic) → lotes más grandes y capacidad de recepción preparada.  
- **SS** crece con la volatilidad de demanda (σ_d) y del **Lead Time** (σ_LT) → servicio por segmento (A alto, B/C moderado) y +1 escalón en eventos.  
- **ROP** se mueve con μ_LT y demanda esperada → **ajuste mensual** con el perfil del modelo.  
- **Lead Time:** controla **P90/P95** por CD×ciudad; **reducir varianza logística** baja SS sin perder servicio.

---

## Inventario: Lecturas de Q*, SS, ROP y Lead Time
> Operativo sin cambiar código: interpreta los resultados/heatmaps existentes.

**Q Óptimo (EOQ)**  
- **Observación:** Q* crece en **sep/dic** por mayor demanda `D`.  
- **Implicación:** elevar tamaño de lote en picos y reducir en valles para minimizar costo de tenencia **H**.

**Stock de Seguridad (SS)**  
- **Observación:** SS es sensible a la **volatilidad diaria (σ_d)** y a la **variabilidad del LT (σ_LT)**.  
- **Implicación:** segmentar niveles de servicio: **A (Hogar/Oficina)** alto, **B/C** moderado; **+1 escalón** de servicio en ventanas de evento.

**Punto de Reorden (ROP)**  
- **Observación:** ROP se desplaza con μ_LT y la demanda esperada.  
- **Implicación:** **ROP dinámico mensual** usando la demanda del modelo para el siguiente periodo; cobertura mínima en días.

**Lead Time**  
- **Observación:** rutas con colas largas (P90/P95) inflan SS sin aumentar ventas.  
- **Implicación:** atacar **varianza logística** (carrier alterno, *expedite*, ventanas de corte) antes que subir SS indefinidamente.

---

## Cierre Metodológico (sin modificar código)

### 1) Validación temporal (*walk-forward*)
- **Esquema:** mensual por categoría; entrenar hasta m-1 y predecir m; 18–24 *folds*.  
- **Métricas:** WAPE, sMAPE, MASE, MAE, R², Bias.  
- **Criterios:** Agregado WAPE ≤ 12%, Bias ∈ [−2%, +2%]. Categorías A ≤ 15%.

**Plantilla — Backtesting (resumen mensual)**  
| Mes      | Categoría | WAPE | sMAPE | MASE | Bias | Observaciones |
|----------|-----------|-----:|------:|-----:|-----:|---------------|
| 2024-09  | Hogar     |      |       |      |      |               |
| 2024-09  | Moda      |      |       |      |      |               |
| …        | …         |      |       |      |      |               |

### 2) Protocolo anti-leakage
- Solo features **exógenas** y disponibles ex-ante; etiquetas basadas en el target entran solo con **rezagos**.

**Matriz de disponibilidad (ejemplo)**  
| Feature             | Fuente     | Disponible en t | Rezago req. | Riesgo leakage | Nota                                    |
|---------------------|------------|-----------------|-------------|----------------|-----------------------------------------|
| Día del Año         | Calendario | Sí              | No          | Bajo           |                                         |
| Producto_Premium    | Catálogo   | Sí              | No          | Bajo           |                                         |
| Descuento_Aplicado  | Pricing    | Sí*             | No          | Medio          | Si no está planificado, usar escenario  |
| BCG_Clasificación   | Derivada y | No              | Sí          | Alto           | No usar en t                            |

### 3) Pronósticos probabilísticos y servicio
- Publicar **P50** y **P10/P90**; cobertura objetivo **P90 ≈ 90%**.  
- Mapa z ↔ cuantiles aproximado: **90% ≈ 1.28**, **95% ≈ 1.65**, **98% ≈ 2.05**.

**Curva de cobertura (plantilla)**  
| Categoría | Nivel | Cobertura esperada | Observada | Gap   |
|-----------|------:|-------------------:|----------:|------:|
| Hogar     | 95%   | 95%                | 92%       | −3 pp |

### 4) Política de inventario (unidades y revisión)
- Horizonte **mensual** recomendado; alinear **D, S, H** a mensual.  
- Fórmulas (texto plano):  
  - **EOQ (Q\*)**: `Q* = sqrt(2*D*S / H)`  
  - **Stock de Seguridad (SS)**: `SS = z * sqrt( (σ_d^2 * LT) + (d^2 * σ_LT^2) )`  
  - **Punto de Reorden (ROP)**: `ROP = μ_d * μ_LT + SS`  
- **Servicio por segmento:** A = 97–98%, B = 95%, C = 92–95%.  
- **Revisión:** mensual (ciclo S&OP) con último año móvil.

**Tabla Q\*/ROP/SS (plantilla)**  
| Periodo | SKU     | Cat  | D(m) | S   | H   | Q*  | μ_d | μ_LT(d) | σ_d | σ_LT | z  | SS  | ROP | Cob.(d) |
|---------|---------|------|-----:|----:|----:|----:|----:|--------:|----:|-----:|---:|----:|----:|--------:|
| 2024-09 | ABC-123 | Hogar|      |     |     |     |     |         |     |      |    |     |     |         |

### 5) Gobierno de Lead Time
- KPIs por **CD×Ciudad×Mes**: μ_LT, σ_LT, **P90**, **P95**.  
- **Alerta**: +20% vs baseline (6m) en σ_LT o P95 → plan de mitigación.

**SLA (plantilla)**  
| Ruta            | μ_LT Obj. | P95 Máx. | Estado | Acción          |
|-----------------|----------:|---------:|--------|-----------------|
| CD-SCL → RM     | 2.5 d     | 4 d      | OK     | –               |

### 6) Diagnóstico de error (operable)
- Descomponer en **volumen**, **mix** y **calendario**; matriz de signo (sub/sobre) por categoría.

**Error por categoría (plantilla)**  
| Categoría | WAPE | Bias | % Sub-Pred. | Nota operativa            |
|-----------|-----:|-----:|------------:|---------------------------|
| Hogar     |      |      |             | +SS mientras estabiliza   |

### 7) Segmentación ABC–XYZ
- **ABC:** por valor anual. **XYZ:** por CV de demanda.  
- Política por celda (ej.): **A-X 98%** (revisión semanal); **C-Z 92%** (revisión mensual).

### 8) Escenarios y sensibilidad
- Escenarios: Base, Promoción, Precio +5%, Evento (sep/dic).  
- Tornado: S, H, μ_LT, σ_LT, σ_d → **frontera costo-servicio**.

### 9) KPIs, S&OP y Champion–Challenger
- **KPIs:** WAPE, Bias, cobertura P90/95; Fill-rate, quiebres, cobertura, rotación, costo inventario; μ_LT, P95_LT.  
- **Cadencia mensual:** Data freeze → Backtest → S&OP (z, Q*, ROP) → Publicación.  
- **Regla:** *challenger* reemplaza si mejora WAPE ≥ **1.5 p.p.** sin empeorar Bias ni cobertura.

### 10) Gobernanza y checklists
- **Paquete mensual:** pronósticos P50/P90, métricas por fold, tablas Q*/ROP/SS, KPIs, LT, acta S&OP.  
- **Checklist antes de publicar:** data freeze ✓, backtesting ✓, KPIs ✓, Q*/ROP/SS ✓, LT vs SLA ✓, acta ✓.

---

## Conclusiones
- **Estacionalidad fuerte** (sep/dic) y patrón intra-semana suave → ROP/SS **dinámicos** y Q* **estacional**.  
- **Modelo enriquecido** mejora el seguimiento y reduce errores en picos.  
- **Inventario por segmento:** A con servicio alto, B/C moderado; **buffers** tácticos donde el modelo sub-predice.  
- **Lead Time**: gestionar **varianza** (σ_LT, P95) es la palanca más eficiente para bajar SS sin perder servicio.

---

## Herramientas Usadas
Python 3.x · pandas · numpy · matplotlib · seaborn · scikit-learn · openpyxl.

---

## Diseño del Código y Automatización Inteligente
Arquitectura modular: `src/codigo_principal.py` orquesta funciones de `src/Funciones.py`.  
1_ Reutilización de funciones para EDA/modelado/gráficos.  
2_ Ejecución rápida por parámetros (columnas, años, filtros).  
3_ Automatización del flujo EDA → modelo → reportes.  
4_ Formato CLP consistente en etiquetas/ejes.  
5_ Flexibilidad para nuevos cortes y escenarios.

---

## Ejecución del Proyecto
```bash
# Ejecutar flujo principal (VS Code / terminal)
python src/codigo_principal.py
# (o)
python src/C_Principal.py
```
> Asegura que `Base_Empresa_BigData_Limpio.xlsx` esté en la **raíz** del repo.

---

## Contacto
Proyecto para portafolio profesional en ciencia de datos aplicada a retail.  
- 📬 lorenzoschiappacase@gmail.com  
- 📎 https://www.linkedin.com/in/lorenzo-brunod-schiappacase-9a10191b9

