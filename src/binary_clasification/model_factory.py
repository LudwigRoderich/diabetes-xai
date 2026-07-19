"""
Fábrica de modelos: instancia y entrena un modelo registrado en
MODEL_CONFIGS aplicando el mecanismo de balanceo de clases correcto
(sample_weight o scale_pos_weight) según su configuración.

Centraliza la lógica porque debe ser idéntica en dos lugares distintos:
- Trainer (entrenamiento real de los modelos finales / folds de CV).
- HyperparameterOptimizer (búsqueda de hiperparámetros con Optuna).

"""
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src.config import MODEL_CONFIGS, get_logger

logger = get_logger(__name__)


def compute_class_weights(y: pd.Series) -> dict:
    """Calcula pesos de clase balanceados (compartido por Trainer y el optimizador)."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def build_and_fit(
    model_tag: str,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    class_weights: dict | None = None,
):
    """
    Instancia y entrena el modelo `model_tag` con `params`, aplicando el
    mecanismo de balanceo de clase declarado en su entrada de MODEL_CONFIGS:
    - uses_scale_pos_weight=True (ej. XGBoost): agrega scale_pos_weight
      al constructor antes de instanciar.
    - uses_sample_weight=True (ej. Decision Tree): pasa sample_weight
      por-instancia en fit().

    `class_weights` puede pasarse precalculado (evita recomputarlo si el
    caller ya lo necesita para otro propósito, ej. logging/almacenamiento);
    si no se pasa, se calcula internamente sobre `y`.
    """
    if model_tag not in MODEL_CONFIGS:
        logger.error(f"Modelo no registrado en MODEL_CONFIGS: '{model_tag}'")
        raise ValueError(f"Modelo desconocido: '{model_tag}'. Disponibles: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model_tag]
    weights = class_weights if class_weights is not None else compute_class_weights(y)
    params = dict(params)

    if cfg["uses_scale_pos_weight"]:
        params["scale_pos_weight"] = weights[0] / weights[1]

    model = cfg["estimator_class"](**params)

    fit_kwargs = {}
    if cfg["uses_sample_weight"]:
        fit_kwargs["sample_weight"] = y.map(weights)

    model.fit(X, y, **fit_kwargs)
    return model