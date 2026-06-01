"""
Evaluación de modelos de clasificación binaria.

Métricas reportadas:
  - PR-AUC  : área bajo la curva precisión-recall (basada en probabilidades).
  - Precision: precisión al umbral indicado.
  - Recall   : exhaustividad al umbral indicado.

El umbral de clasificación es un parámetro explícito; por defecto usa
DEFAULT_THRESHOLD definido en config.py.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from typing import Optional

from config import DEFAULT_THRESHOLD, THRESHOLD_SEARCH_MIN, THRESHOLD_SEARCH_MAX, THRESHOLD_STEPS


class Evaluator:
    """
    Calcula y agrega métricas de evaluación para modelos de clasificación
    binaria que exponen predict_proba().

    Parameters
    ----------
    threshold : float, optional
        Umbral de clasificación para convertir probabilidades en etiquetas.
        Si es None, usa DEFAULT_THRESHOLD de config.py.

    Attributes
    ----------
    threshold : float
        Umbral activo en la instancia.
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLD

    # ------------------------------------------------------------------
    def evaluate(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        """
        Evalúa un modelo sobre el conjunto indicado.

        Parameters
        ----------
        model : clasificador sklearn-compatible
            Debe exponer predict_proba(). Se usa la columna de probabilidad
            de la clase positiva (índice 1).
        X : pd.DataFrame
            Features del conjunto de evaluación, ya escaladas.
        y : pd.Series
            Etiquetas verdaderas.

        Returns
        -------
        dict con claves: prauc, precision, recall, threshold.
        """
        proba  = self._get_proba(model, X)
        labels = self._proba_to_labels(proba, self.threshold)

        return {
            "prauc"    : average_precision_score(y, proba),
            "precision": precision_score(y, labels, zero_division=0),
            "recall"   : recall_score(y, labels, zero_division=0),
            "threshold": self.threshold,
        }

    # ------------------------------------------------------------------
    def optimize_threshold(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        strategy: str = "f1",
        min_recall: Optional[float] = None,
    ) -> float:
        """
        Busca el umbral óptimo sobre el conjunto de tuning mediante un
        barrido uniforme en [THRESHOLD_SEARCH_MIN, THRESHOLD_SEARCH_MAX].

        Parameters
        ----------
        model : clasificador sklearn-compatible
            Modelo ya entrenado sobre X_train completo.
        X : pd.DataFrame
            Features del conjunto de tuning (el 10 % reservado), ya escaladas.
        y : pd.Series
            Etiquetas verdaderas del conjunto de tuning.
        strategy : str
            Criterio de optimización. Opciones:
              'f1'     : maximiza F1 entre Precision y Recall.
              'recall' : maximiza Recall sujeto a min_recall si se indica;
                         en la práctica maximiza Recall puro cuando
                         min_recall es None.
              'prauc'  : selecciona el umbral que maximiza PR-AUC local
                         (equivale a encontrar el punto de mayor
                         Precision * Recall en la curva).
        min_recall : float, optional
            Restricción mínima de Recall. Solo aplica cuando strategy='f1'
            o strategy='prauc' para filtrar umbrales que no la cumplan.
            Si ningún umbral la cumple, se relaja automáticamente y se
            emite una advertencia.

        Returns
        -------
        float
            Umbral óptimo. Actualiza self.threshold como efecto secundario.
        """
        proba      = self._get_proba(model, X)
        thresholds = np.linspace(
            THRESHOLD_SEARCH_MIN,
            THRESHOLD_SEARCH_MAX,
            THRESHOLD_STEPS,
        )

        best_threshold = self.threshold
        best_score     = -np.inf

        for t in thresholds:
            labels    = self._proba_to_labels(proba, t)
            precision = precision_score(y, labels, zero_division=0)
            recall    = recall_score(y, labels, zero_division=0)

            # aplicar restricción de recall mínimo si se especifica
            if min_recall is not None and recall < min_recall:
                continue

            score = self._compute_strategy_score(
                strategy, float(precision), float(recall), proba, labels, y
            )

            if score > best_score:
                best_score     = score
                best_threshold = t

        # si la restricción min_recall dejó vacío el espacio de búsqueda,
        # se relaja y se repite sin restricción
        if best_score == -np.inf:
            print(
                f"[Evaluator] Advertencia: ningún umbral cumple min_recall="
                f"{min_recall:.2f}. Se relaja la restricción."
            )
            return self.optimize_threshold(model, X, y, strategy=strategy, min_recall=None)

        self.threshold = float(best_threshold)
        return self.threshold

    # ------------------------------------------------------------------
    def evaluate_curve(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        """
        Devuelve los vectores completos de la curva Precision-Recall para
        su uso en visualizaciones.

        Returns
        -------
        dict con claves: precision_curve, recall_curve, thresholds, prauc.
        """
        proba                         = self._get_proba(model, X)
        precision_c, recall_c, thresh = precision_recall_curve(y, proba)

        return {
            "precision_curve": precision_c,
            "recall_curve"   : recall_c,
            "thresholds"     : thresh,
            "prauc"          : average_precision_score(y, proba),
        }

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    @staticmethod
    def _get_proba(model, X: pd.DataFrame) -> np.ndarray:
        """Extrae la probabilidad de la clase positiva."""
        return model.predict_proba(X)[:, 1]

    @staticmethod
    def _proba_to_labels(proba: np.ndarray, threshold: float) -> np.ndarray:
        """Convierte probabilidades a etiquetas binarias según el umbral."""
        return (proba >= threshold).astype(int)

    @staticmethod
    def _compute_strategy_score(
        strategy: str,
        precision: float,
        recall: float,
        proba: np.ndarray,
        labels: np.ndarray,
        y: pd.Series,
    ) -> float:
        """
        Calcula el score de optimización según la estrategia indicada.

        Returns
        -------
        float : score a maximizar.
        """
        if strategy == "f1":
            denom = precision + recall
            return (2 * precision * recall / denom) if denom > 0 else 0.0

        if strategy == "recall":
            return recall

        if strategy == "prauc":
            # producto Precision * Recall como proxy local del área
            return precision * recall

        raise ValueError(
            f"Estrategia '{strategy}' no reconocida. "
            "Opciones válidas: 'f1', 'recall', 'prauc'."
        )