# Pipeline de Clasificación Binaria — Entorno XAI

Arquitectura modular de Machine Learning orientada a datos tabulares desbalanceados. El objetivo central no es únicamente el rendimiento predictivo, sino garantizar un preprocesamiento estricto y libre de fuga de datos (*data leakage*) que soporte análisis posteriores de Explicabilidad (XAI).

Este repositorio es la base computacional del proyecto de seminario de investigación *Interpretación y Validación de Técnicas XAI en Predicción de Diabetes* (MM-700, UNAH). El reporte formal, que incluye los análisis de estabilidad SHAP/LIME y las correlaciones de Spearman, se publicará en este mismo repositorio en fases posteriores.

---

## Estructura del proyecto

```
.
├── data/
│   ├── diabetes_binary_health_indicators_BRFSS2015.csv
│   └── diabetes_binary_health_indicators_BRFSS2015_no_outliners.csv
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── optimizer.py
│   ├── persistence.py
│   ├── visualizer.py
│   └── __init__.py
├── outputs/
│   ├── figures/
│   ├── logs/
│   ├── models/
│   ├── results/
│   └── scalers/
└── main.py
```

---

## Módulos en `src/`

### `config.py`

Punto centralizado de configuración. Ningún módulo debe tener valores numéricos o rutas *hardcodeadas*; todo se importa desde aquí. Modificar este archivo es el primer paso al adaptar el proyecto a un nuevo problema.

**Constantes relevantes:**

| Categoría | Constante | Valor por defecto |
|---|---|---|
| Partición | `TRAIN_SIZE / VAL_SIZE / TUNE_SIZE` | `0.70 / 0.20 / 0.10` |
| CV | `N_FOLDS` | `5` |
| Variable objetivo | `TARGET_COL` | `"Diabetes_binary"` |
| Variables continuas | `CONTINUOUS_COLS` | `["BMI", "MentHlth", "PhysHlth"]` |
| Variables ordinales | `ORDINAL_COLS` | `["Age", "Education", "Income", "GenHlth"]` |
| Umbral por defecto | `DEFAULT_THRESHOLD` | `0.5` |
| Barrido de umbral | `THRESHOLD_SEARCH_MIN/MAX/STEPS` | `0.01 / 0.99 / 120` |
| Semillas | `GLOBAL_SEED`, `SPLIT_SEED`, `CV_SEED`, `DT_SEED`, `XGBOOST_SEED` | Fijas e independientes por etapa |

Los parámetros base de los modelos (`DT_BASE_PARAMS`, `XGBOOST_BASE_PARAMS`) también residen aquí y son los que utiliza `Trainer` cuando no se le pasan parámetros externos.

---

### `data_loader.py` — clase `DataLoader`

Gestiona la carga del CSV y la partición estratificada en tres conjuntos disjuntos.

```python
DataLoader(use_clean: bool = False)
```

- `use_clean=True`: carga el dataset sin outliers (`_no_outliners.csv`).
- `use_clean=False`: carga el dataset original completo.

**Métodos:**

`load() -> DataLoader`
Carga el CSV desde la ruta definida en `config.py`. Devuelve `self` para encadenamiento.

`split() -> DataLoader`
Particiona estratificadamente en tres subconjuntos preservando la proporción de la clase objetivo. El proceso es secuencial:
1. Se separa el conjunto de *tuning* (10 %) del total.
2. Del 90 % restante, se separa validación (≈22.2 % del residuo, equivalente al 20 % del total).
3. El remanente constituye entrenamiento (70 % del total).

Popula los atributos `X_train`, `X_val`, `X_tune`, `y_train`, `y_val`, `y_tune`.

`get_cv_folds() -> list[tuple[ndarray, ndarray]]`
Genera los índices de los `N_FOLDS` folds estratificados sobre `X_train`. Devuelve una lista de tuplas `(idx_train_fold, idx_val_fold)` con índices posicionales.

`class_proportions(y: pd.Series | None = None) -> dict`
Calcula la proporción de cada clase. Si `y` es `None`, usa `y_train`.

---

### `preprocessor.py` — clase `Preprocessor`

Aplica escalado estándar de forma selectiva según el tipo de variable y el contexto de uso (modelo o XAI). Esta separación es fundamental: los modelos predictivos nunca reciben variables ordinales escaladas, mientras que los métodos XAI pueden recibirlas si es necesario para sus cálculos de distancia.

```python
Preprocessor(fold_id: int | str, dataset_tag: str = "full")
```

- `fold_id`: identificador del fold o `"final"` para el modelo definitivo.
- `dataset_tag`: `"full"` o `"clean"`. Se usa en el nombre del archivo serializado.

**Métodos:**

`fit(X: pd.DataFrame) -> Preprocessor`
Ajusta dos `StandardScaler` internos sobre el conjunto de entrenamiento del fold:
- `_scaler_model`: ajustado sobre `CONTINUOUS_COLS` únicamente.
- `_scaler_xai`: ajustado sobre `CONTINUOUS_COLS + ORDINAL_COLS`.

`transform(X: pd.DataFrame, xai_mode: bool = False) -> pd.DataFrame`
Devuelve una copia del DataFrame con las columnas correspondientes escaladas. Las variables binarias permanecen intactas.
- `xai_mode=False`: escala solo `CONTINUOUS_COLS` (uso en entrenamiento y evaluación).
- `xai_mode=True`: escala `CONTINUOUS_COLS + ORDINAL_COLS` (uso en métodos XAI).

`fit_transform(X, xai_mode=False) -> pd.DataFrame`
Combina `fit()` y `transform()` en un solo paso. Se usa exclusivamente sobre el subconjunto de entrenamiento del fold.

`save() -> Path`
Serializa ambos scalers en `outputs/scalers/scaler_{dataset_tag}_fold{fold_id}.pkl`. Lanza `RuntimeError` si el objeto no ha sido ajustado.

`Preprocessor.load(fold_id, dataset_tag) -> Preprocessor` *(classmethod)*
Reconstruye una instancia desde disco. El objeto resultante tiene `_is_fitted = True` y está listo para llamar a `transform()`.

---

### `trainer.py` — clase `Trainer`

Orquesta el ciclo completo de entrenamiento: validación cruzada y entrenamiento final sobre `X_train` completo.

```python
Trainer(
    loader: DataLoader,
    dataset_tag: str = "full",
    dt_params: dict | None = None,
    xgb_params: dict | None = None,
)
```

- `dt_params` y `xgb_params`: si son `None`, se usan `DT_BASE_PARAMS` y `XGBOOST_BASE_PARAMS` de `config.py`. Pasa un diccionario externo para usar hiperparámetros optimizados por Optuna.

**Métodos:**

`run_cv() -> Trainer`
Ejecuta `N_FOLDS` iteraciones. Por cada fold:
1. Separa los índices de entrenamiento y validación del fold.
2. Instancia un `Preprocessor`, lo ajusta sobre el subconjunto de entrenamiento del fold y lo serializa en disco.
3. Calcula pesos de clase balanceados independientemente para ese fold.
4. Entrena `DecisionTreeClassifier` y `XGBClassifier`.
5. Evalúa ambos modelos sobre el subconjunto de validación del fold con umbral por defecto (`τ = 0.5`).
6. Almacena el resultado en `self.fold_results`.

El cálculo de pesos por fold —no sobre el total del entrenamiento— evita que la distribución global contamine el balance dentro de cada pliegue.

`train_final() -> Trainer`
Reentrena ambos modelos sobre el 100 % de `X_train`, con un `Preprocessor` nuevo ajustado sobre ese mismo conjunto (serializado como `fold_id="final"`). Este es el modelo que se usa en la fase XAI.

`get_cv_summary() -> pd.DataFrame`
Agrega las métricas de todos los folds en un DataFrame. Columnas: `fold`, `dataset_tag`, `{modelo}_{métrica}` (e.g., `dt_prauc`, `xgb_recall`).

`_compute_weights(y: pd.Series) -> dict`
Método estático interno. Aplica la fórmula `w_i = n / (k * n_i)` de scikit-learn. Para XGBoost, `_train_xgb()` deriva `scale_pos_weight = w_0 / w_1` a partir de este resultado.

---

### `evaluator.py` — clase `Evaluator`

Calcula métricas de rendimiento y optimiza el umbral de clasificación.

```python
Evaluator(threshold: float | None = None)
```

- Si `threshold` es `None`, usa `DEFAULT_THRESHOLD` de `config.py`.

**Métodos:**

`evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict`
Evalúa el modelo sobre el conjunto indicado. El modelo debe exponer `predict_proba()`.

Devuelve:
```python
{
    "prauc"    : float,  # área bajo la curva Precision-Recall (basada en probabilidades)
    "precision": float,  # al umbral activo
    "recall"   : float,  # al umbral activo
    "threshold": float,  # umbral usado
}
```

`optimize_threshold(model, X, y, strategy: str = "f1", min_recall: float | None = None) -> float`
Realiza un barrido uniforme de 120 puntos en `[0.01, 0.99]` y selecciona el umbral que maximiza el criterio indicado.

- `strategy="f1"`: maximiza `2 * P * R / (P + R)`.
- `strategy="recall"`: maximiza Recall puro.
- `strategy="prauc"`: maximiza el producto `P * R` como proxy local del área.
- `min_recall`: restricción de Recall mínimo. Si ningún umbral la cumple, se relaja automáticamente con advertencia.

Actualiza `self.threshold` como efecto secundario y devuelve el umbral óptimo.

`evaluate_curve(model, X, y) -> dict`
Devuelve los vectores completos de la curva PR para visualización.
```python
{
    "precision_curve": ndarray,
    "recall_curve"   : ndarray,
    "thresholds"     : ndarray,
    "prauc"          : float,
}
```

---

### `optimizer.py`

Módulo de optimización bayesiana de hiperparámetros usando Optuna. Se activa exclusivamente mediante el flag `--optimize` en `main.py`.

`optimize_cv(model_name: str, dataset_tag: str, n_trials: int) -> dict`
Instancia un estudio Optuna que evalúa combinaciones de hiperparámetros ejecutando el CV completo en cada trial. Implementa parada temprana por convergencia y un sistema anti-podas que previene que un fold fallido descarte un trial potencialmente bueno.

Devuelve el diccionario de hiperparámetros óptimos encontrados, que puede pasarse directamente a `Trainer(dt_params=...)` o `Trainer(xgb_params=...)`.

---

### `persistence.py` — clase `Persistence`

Colección de métodos estáticos para I/O de artefactos. Mantiene la lógica de serialización separada del resto de la lógica de negocio.

| Método | Formato | Ruta de salida |
|---|---|---|
| `save_model(model, model_name, fold_id, dataset_tag)` | `.pkl` (joblib) | `outputs/models/{model_name}_{dataset_tag}_fold{fold_id}.pkl` |
| `load_model(model_name, fold_id, dataset_tag)` | — | — |
| `save_fold_metrics(metrics, fold_id, dataset_tag)` | `.json` | `outputs/results/metrics_{dataset_tag}_fold{fold_id}.json` |
| `save_cv_summary(summary, dataset_tag)` | `.csv` | `outputs/results/cv_summary_{dataset_tag}.csv` |
| `save_final_results(results, dataset_tag)` | `.json` | `outputs/results/final_results_{dataset_tag}.json` |
| `list_saved_models(dataset_tag=None)` | — | lista de `Path` |
| `list_saved_results(dataset_tag=None)` | — | lista de `Path` |

---

### `visualizer.py`

Genera y guarda figuras en `outputs/figures/`. Ninguna función llama a `plt.show()`; el flujo es completamente no interactivo. Todas devuelven la `Path` del archivo guardado.

| Función | Salida |
|---|---|
| `plot_fold_metrics(summary, dataset_tag, metrics=None)` | `fold_metrics_{dataset_tag}.png` — barras agrupadas por modelo y fold |
| `plot_fold_metrics_optimal(summary, dataset_tag, metrics=None)` | `fold_metrics_{dataset_tag}_optimal.png` — igual, con umbral óptimo |
| `plot_pr_curves(curve_data, dataset_tag)` | `pr_curves_{dataset_tag}.png` — curvas PR de DT y XGBoost superpuestas |
| `plot_dataset_comparison(summary_full, summary_clean, metrics=None)` | `dataset_comparison.png` — comparación de métricas promedio full vs. clean |
| `plot_dataset_comparison_optimal(summary_full, summary_clean, metrics=None)` | `dataset_comparison_optimal.png` — igual, con umbral óptimo |
| `plot_class_weights(fold_results, dataset_tag)` | `class_weights_{dataset_tag}.png` — evolución de pesos de clase por fold |

El parámetro `metrics` acepta cualquier subconjunto de `["prauc", "precision", "recall"]`. Por defecto incluye los tres.

---

### `main.py`

Punto de entrada. Amarra todos los módulos anteriores en un flujo secuencial de 8 pasos, gestiona la CLI y evita recomputación innecesaria verificando artefactos existentes antes de ejecutar.

**Flujo de `run_pipeline(dataset_tag)`:**

1. Carga y partición estratificada (`DataLoader.load().split()`).
2. Validación cruzada de `N_FOLDS` folds (`Trainer.run_cv()`).
3. Serialización de métricas por fold y resumen del CV (`Persistence`).
4. Entrenamiento del modelo final sobre `X_train` completo (`Trainer.train_final()`).
5. Evaluación sobre `X_val` con umbral por defecto (`τ = 0.5`).
6. Optimización del umbral sobre `X_tune` (`Evaluator.optimize_threshold()`).
7. Evaluación final sobre `X_tune` y `X_val` con umbral óptimo.
8. Serialización de resultados finales y recálculo del CV summary con umbral óptimo.

`artifacts_exist(dataset_tag) -> bool`
Comprueba si `cv_summary_{tag}.csv` y `final_results_{tag}.json` existen en `outputs/results/`. Si es `True` y no se usa `--force-retrain`, se omite el pipeline y se cargan los artefactos desde disco.

---

## Directorio `outputs/`

### `figures/`

Almacena los PNG generados por `visualizer.py`. Son los únicos artefactos pensados para inserción directa en reportes.

- `fold_metrics_{tag}.png`: métricas (PR-AUC, Precision, Recall) por fold y modelo con umbral por defecto. Permite detectar varianza entre folds.
- `fold_metrics_{tag}_optimal.png`: igual, con el umbral optimizado sobre `X_tune`.
- `pr_curves_{tag}.png`: curvas PR completas de DT y XGBoost sobre `X_val`. Útil para comparar el tradeoff Precision-Recall a lo largo de todos los umbrales posibles.
- `dataset_comparison.png` y `dataset_comparison_optimal.png`: comparación de métricas promedio entre el dataset completo y el depurado. Permite determinar si la eliminación de outliers introduce o no una mejora significativa.
- `class_weights_{tag}.png`: evolución de los pesos asignados a cada clase a lo largo de los folds. La variación entre folds debe ser mínima; una variación alta indica que la estratificación no funcionó correctamente.

### `logs/`

Registro completo de ejecución. `pipeline_execution.log` incluye timestamp, nombre del archivo y número de línea para cada evento. Los logs crudos de Optuna se redirigen aquí para no saturar la terminal durante optimizaciones largas.

### `models/`

Archivos `.pkl` (joblib) con los modelos serializados. Convención de nombres:

```
{model_name}_{dataset_tag}_fold{fold_id}.pkl
```

Ejemplos: `dt_clean_fold2.pkl`, `xgb_full_foldfinal.pkl`. El modelo `foldfinal` es el que consume la fase XAI, ya que fue entrenado sobre el 100 % de `X_train`.

### `results/`

Contiene tres tipos de archivos:

- `metrics_{dataset_tag}_fold{N}.json`: métricas de DT y XGBoost para el fold N, con umbral por defecto. Útil para auditoría por fold.
- `cv_summary_{dataset_tag}.csv`: agregado de todos los folds en una tabla. Importable directamente en R o Excel.
- `final_results_{dataset_tag}.json`: resultados del modelo final sobre `X_val` y `X_tune`, umbral óptimo encontrado y pesos de clase. Estructura:

```json
{
  "dataset_tag": "clean",
  "optimal_thresholds": {"dt": 0.35, "xgb": 0.28},
  "val_metrics": {"dt": {...}, "xgb": {...}},
  "tune_metrics": {"dt": {...}, "xgb": {...}},
  "class_weights": {"0": 0.578, "1": 3.689}
}
```

### `scalers/`

Archivos `.pkl` con los `StandardScaler` serializados por `Preprocessor`. Convención de nombres:

```
scaler_{dataset_tag}_fold{fold_id}.pkl
```

Cada archivo contiene dos scalers independientes (`scaler_model` y `scaler_xai`). El archivo `foldifinal` es el que debe usarse para transformar datos nuevos en producción o para alimentar SHAP y LIME correctamente, ya que corresponde al modelo entrenado sobre el 100 % de `X_train`.

---

## Interfaz de línea de comandos

Todos los argumentos son opcionales. Sin argumentos, ejecuta el pipeline completo sobre ambos datasets con parámetros por defecto.

```bash
python main.py [--dataset {full,clean,both}] [--model {dt,xgb,both}]
               [--force-retrain] [--skip-cv] [--thresh-metric {prauc,f1,recall}]
               [--optimize] [--optimize-model {dt,xgb,both}]
               [--plot-history] [--no-show]
```

### `--dataset {full | clean | both}`

Define el dataset sobre el que se ejecuta el flujo.

- `full`: dataset original completo (253 680 instancias).
- `clean`: dataset sin outliers detectados por Isolation Forest (≈248 461 instancias).
- `both` *(defecto)*: ejecuta el flujo secuencialmente sobre ambos y genera la figura comparativa.

### `--model {dt | xgb | both}`

Restringe el entrenamiento a un modelo específico.

- `dt`: solo árbol de decisión.
- `xgb`: solo XGBoost.
- `both` *(defecto)*: ambos modelos.

### `--force-retrain`

Ignora cualquier artefacto existente en `outputs/results/` y ejecuta el pipeline completo desde cero. Sin este flag, si `cv_summary_{tag}.csv` y `final_results_{tag}.json` ya existen, el pipeline carga desde disco y salta directamente a la generación de visualizaciones.

### `--skip-cv`

Omite la validación cruzada. Entrena directamente sobre el 70 % de entrenamiento completo y evalúa sobre el 20 % de validación. Útil para pruebas rápidas de concepto o cuando los hiperparámetros ya fueron determinados y solo se necesita el modelo final.

### `--thresh-metric {prauc | f1 | recall}`

Define el criterio de optimización del umbral sobre `X_tune`.

- `prauc` *(defecto)*: maximiza `Precision * Recall` (proxy del área bajo la curva PR).
- `f1`: maximiza F1-score.
- `recall`: maximiza Recall puro (útil si la prioridad es capturar todos los positivos a costa de precisión).

### `--optimize`

Activa el modo de optimización bayesiana con Optuna. En este modo no se ejecuta el pipeline de entrenamiento estándar; en su lugar se lanza la búsqueda intensiva de hiperparámetros. Requiere `--optimize-model`.

### `--optimize-model {dt | xgb | both}`

Especifica qué modelo optimizar cuando se usa `--optimize`. Ignorado si `--optimize` no está presente.

### `--plot-history`

Genera la gráfica de evolución del PR-AUC a lo largo de los trials de Optuna (gráfica de escaleras). Requiere que existan logs de Optuna en `outputs/logs/`.

### `--no-show`

Suprime la apertura de ventanas emergentes de matplotlib. Recomendado para ejecuciones automatizadas en scripts `.bat` o pipelines de shell.

---

## Pipelines de ejemplo

### Pipeline 1: Optimización bayesiana sobre el dataset depurado

Caso de uso: determinar los hiperparámetros óptimos de XGBoost antes de ejecutar el pipeline formal. Se ejecuta en segundo plano (preferiblemente de noche) y serializa el mejor conjunto de parámetros en `outputs/results/`.

```bash
python main.py --optimize --optimize-model xgb --dataset clean
```

**Pasos internos:**
1. `DataLoader` carga el dataset depurado y ejecuta la partición 70/20/10.
2. Optuna instancia un estudio con el espacio de búsqueda definido en `config.py`.
3. Por cada trial, `Trainer.run_cv()` ejecuta los 5 folds con el conjunto de hiperparámetros propuesto. El PR-AUC promedio del CV es el valor objetivo del trial.
4. Parada temprana por convergencia cuando la mejora marginal cae por debajo del umbral configurado.
5. El mejor conjunto de hiperparámetros se serializa en `outputs/results/best_params_xgb_clean.json`.
6. Si se pasa `--plot-history`, se genera la curva de mejora acumulada por trial.

---

### Pipeline 2: Evaluación rápida con hiperparámetros ya conocidos

Caso de uso: verificar el rendimiento de los modelos con hiperparámetros ya optimizados, omitiendo el CV para obtener las métricas y gráficas finales en el menor tiempo posible. Este pipeline fue el utilizado para generar las figuras del reporte de avance.

```bash
python main.py --dataset both --model both --thresh-metric prauc --force-retrain --skip-cv
```

**Pasos internos:**
1. `--force-retrain` garantiza que no se cargue ningún artefacto previo aunque existan.
2. `DataLoader` carga y particiona ambas versiones del dataset (full y clean).
3. `--skip-cv` omite `Trainer.run_cv()`; se salta directamente a `Trainer.train_final()` para DT y XGBoost sobre cada dataset.
4. `Preprocessor` se ajusta sobre `X_train` completo y se serializa como `foldfinal`.
5. Se evalúa sobre `X_val` con `τ = 0.5`.
6. `Evaluator.optimize_threshold()` busca el umbral óptimo sobre `X_tune` maximizando `Precision * Recall` (`strategy="prauc"`).
7. Se evalúa con el umbral óptimo sobre `X_tune` y `X_val`.
8. Se generan `pr_curves_full.png`, `pr_curves_clean.png` y `dataset_comparison.png`.

**Advertencia:** al omitir el CV, las métricas reportadas corresponden a una sola partición y no tienen estimación de varianza. No usar para reportes formales.

---

### Pipeline 3: Ejecución académica completa con CV

Caso de uso: evaluación formal para reporte. Ejecuta el CV de 5 folds para el árbol de decisión sobre el dataset completo, maximizando F1 en la búsqueda del umbral, y sobreescribe cualquier resultado anterior.

```bash
python main.py --dataset full --model dt --thresh-metric f1 --force-retrain
```

**Pasos internos:**
1. `DataLoader` carga el dataset completo y ejecuta la partición estratificada 70/20/10.
2. `Trainer.run_cv()` ejecuta 5 folds. En cada fold:
   a. `Preprocessor.fit_transform()` sobre el subconjunto de entrenamiento del fold.
   b. `Preprocessor.transform()` (sin reajuste) sobre el subconjunto de validación del fold.
   c. Cálculo de pesos de clase balanceados para ese fold.
   d. Entrenamiento de `DecisionTreeClassifier` con `sample_weight`.
   e. `Evaluator.evaluate()` sobre el subconjunto de validación con `τ = 0.5`.
   f. Serialización del modelo y del scaler del fold.
3. `Persistence.save_cv_summary()` guarda el DataFrame con las 5 filas de métricas.
4. `Trainer.train_final()` reentrena el árbol sobre el 100 % de `X_train`.
5. `Evaluator.optimize_threshold()` busca el umbral que maximiza F1 sobre `X_tune`.
6. Evaluación final sobre `X_tune` y `X_val` con el umbral óptimo encontrado.
7. Se generan `fold_metrics_full.png`, `fold_metrics_full_optimal.png`, `class_weights_full.png` y `pr_curves_full.png`.

---

### Pipeline 4: Carga desde artefactos existentes

Caso de uso: regenerar visualizaciones sin reentrenar. Si los artefactos de una ejecución previa están en disco, el pipeline los carga directamente.

```bash
python main.py --dataset both
```

**Pasos internos:**
1. `artifacts_exist("full")` y `artifacts_exist("clean")` comprueban la existencia de `cv_summary_{tag}.csv` y `final_results_{tag}.json`.
2. Si ambos existen, se llama a `load_pipeline_results()` en lugar de `run_pipeline()`.
3. Se cargan `cv_summary` y `final_results` desde disco mediante `Persistence`.
4. Se generan todas las visualizaciones a partir de los datos cargados.

**Nota:** las visualizaciones que requieren acceso al `Trainer` o al `DataLoader` en memoria (curvas PR, pesos de clase) no están disponibles en este modo, ya que esos objetos no se persisten en disco.

---

## Adaptación a otros proyectos

El pipeline no está acoplado al problema de diabetes. Para usarlo con un dataset tabular diferente:

1. Coloca los archivos CSV en `data/`.
2. En `config.py`:
   - Cambia `TARGET_COL` al nombre de la variable objetivo.
   - Actualiza `CONTINUOUS_COLS` con las variables numéricas continuas que deben estandarizarse.
   - Actualiza `ORDINAL_COLS` con las variables ordinales que los métodos XAI recibirán escaladas.
   - Ajusta `TRAIN_SIZE`, `VAL_SIZE` y `TUNE_SIZE` si la proporción 70/20/10 no es adecuada.
   - Ajusta `N_FOLDS` según el tamaño del dataset.
3. Si el desbalance es distinto, los pesos se recalculan automáticamente por fold sin intervención adicional.
4. Si el espacio de búsqueda de Optuna no es apropiado para el nuevo problema, ajústalo en `optimizer.py`.

La lógica de CV estratificado, cálculo de pesos, serialización de artefactos y generación de visualizaciones se adapta automáticamente a los nuevos datos.

---

## Contexto de investigación

Este repositorio soporta el proyecto *Interpretación y Validación de Técnicas XAI en Predicción de Diabetes* (MM-700, UNAH). El trabajo evalúa la validez y estabilidad de cuatro métodos XAI post-hoc — Global Surrogate Tree, Permutation Feature Importance, LIME y SHAP — aplicados al modelo XGBoost final entrenado aquí.

La validez se mide mediante correlación de Spearman entre el ranking de importancia de cada método XAI y el ranking intrínseco del árbol de decisión (ganancia de información acumulada). La estabilidad local se analiza agrupando instancias con distancia de Gower y k-medoids, comparando los vectores de explicación del medoide de cada clúster contra los 5 puntos más cercanos y los 5 más lejanos.

El reporte formal con los resultados de estas fases se publicará en este repositorio al concluir las semanas 5–10 del cronograma.